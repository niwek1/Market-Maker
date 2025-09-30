"""
Kill switch implementation for emergency trading halt.

This module provides multiple kill switch mechanisms:
- Manual kill switch activation
- Automatic kill switch triggers based on risk conditions
- Global trading halt
- Graceful shutdown procedures
- Emergency order cancellation
"""

import asyncio
import time
from enum import Enum
from typing import List, Callable, Optional, Dict, Any
from dataclasses import dataclass

from mm_bot.logging import LoggerMixin


class KillSwitchTrigger(Enum):
    """Kill switch trigger types."""
    MANUAL = "manual"
    DRAWDOWN = "drawdown"
    POSITION_LIMIT = "position_limit"
    ORDER_RATE = "order_rate"
    CONNECTION_LOSS = "connection_loss"
    SYSTEM_ERROR = "system_error"
    EXTERNAL_SIGNAL = "external_signal"


class KillSwitchStatus(Enum):
    """Kill switch status."""
    ACTIVE = "active"      # Normal operation
    TRIGGERED = "triggered"  # Kill switch activated
    SHUTTING_DOWN = "shutting_down"  # In shutdown process
    SHUTDOWN = "shutdown"   # Fully shut down


@dataclass
class KillSwitchEvent:
    """Kill switch activation event."""
    trigger: KillSwitchTrigger
    reason: str
    timestamp: int
    metadata: Dict[str, Any]
    auto_triggered: bool = True


class KillSwitch(LoggerMixin):
    """
    Emergency kill switch for trading operations.
    
    Features:
    - Multiple trigger mechanisms
    - Graceful shutdown procedures
    - Emergency order cancellation
    - Configurable auto-triggers
    - Manual override capabilities
    - Audit trail of all activations
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize kill switch.
        
        Args:
            enabled: Whether kill switch is enabled
        """
        super().__init__()
        self.enabled = enabled
        self.status = KillSwitchStatus.ACTIVE
        
        # Trigger configuration
        self.auto_triggers_enabled = True
        self.trigger_conditions = {
            KillSwitchTrigger.DRAWDOWN: {"threshold": 10.0, "enabled": True},
            KillSwitchTrigger.POSITION_LIMIT: {"threshold": 1.0, "enabled": True},
            KillSwitchTrigger.ORDER_RATE: {"threshold": 100, "enabled": True},
            KillSwitchTrigger.CONNECTION_LOSS: {"timeout_seconds": 30, "enabled": True},
            KillSwitchTrigger.SYSTEM_ERROR: {"error_count": 5, "enabled": True},
        }
        
        # Callbacks for different stages
        self.pre_shutdown_callbacks: List[Callable[[], None]] = []
        self.shutdown_callbacks: List[Callable[[], None]] = []
        self.post_shutdown_callbacks: List[Callable[[], None]] = []
        
        # Event tracking
        self.activation_history: List[KillSwitchEvent] = []
        self.last_activation: Optional[KillSwitchEvent] = None
        
        # State tracking
        self.connection_last_seen = time.time()
        self.error_count = 0
        self.error_window_start = time.time()
        
        # Shutdown tracking
        self.shutdown_start_time: Optional[float] = None
        self.shutdown_tasks: List[asyncio.Task] = []
        
        self.logger.info(
            "Initialized kill switch",
            enabled=enabled,
            auto_triggers_enabled=self.auto_triggers_enabled
        )
    
    def trigger(
        self,
        trigger_type: KillSwitchTrigger,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
        auto_triggered: bool = False
    ) -> bool:
        """
        Trigger the kill switch.
        
        Args:
            trigger_type: Type of trigger
            reason: Human-readable reason
            metadata: Additional context data
            auto_triggered: Whether this was an automatic trigger
            
        Returns:
            True if kill switch was activated
        """
        if not self.enabled:
            self.logger.warning(
                "Kill switch trigger ignored (disabled)",
                trigger_type=trigger_type.value,
                reason=reason
            )
            return False
        
        if self.status != KillSwitchStatus.ACTIVE:
            self.logger.warning(
                "Kill switch already triggered",
                current_status=self.status.value,
                trigger_type=trigger_type.value
            )
            return False
        
        # Create event
        event = KillSwitchEvent(
            trigger=trigger_type,
            reason=reason,
            timestamp=int(time.time() * 1000),
            metadata=metadata or {},
            auto_triggered=auto_triggered
        )
        
        # Record event
        self.activation_history.append(event)
        self.last_activation = event
        
        # Update status
        self.status = KillSwitchStatus.TRIGGERED
        
        self.logger.critical(
            "KILL SWITCH ACTIVATED",
            trigger_type=trigger_type.value,
            reason=reason,
            auto_triggered=auto_triggered,
            metadata=metadata
        )
        
        # Start shutdown process
        asyncio.create_task(self._execute_shutdown())
        
        return True
    
    def manual_trigger(self, reason: str = "Manual activation") -> bool:
        """Manually trigger the kill switch."""
        return self.trigger(
            KillSwitchTrigger.MANUAL,
            reason,
            auto_triggered=False
        )
    
    def check_drawdown_trigger(self, current_drawdown_pct: float) -> bool:
        """
        Check if drawdown trigger should activate.
        
        Args:
            current_drawdown_pct: Current drawdown percentage
            
        Returns:
            True if trigger was activated
        """
        if not self.auto_triggers_enabled:
            return False
        
        trigger_config = self.trigger_conditions[KillSwitchTrigger.DRAWDOWN]
        if not trigger_config["enabled"]:
            return False
        
        if current_drawdown_pct >= trigger_config["threshold"]:
            return self.trigger(
                KillSwitchTrigger.DRAWDOWN,
                f"Drawdown exceeded threshold: {current_drawdown_pct:.2f}% >= {trigger_config['threshold']:.2f}%",
                metadata={"drawdown_pct": current_drawdown_pct, "threshold": trigger_config["threshold"]},
                auto_triggered=True
            )
        
        return False
    
    def check_position_trigger(self, position_utilization: float) -> bool:
        """
        Check if position limit trigger should activate.
        
        Args:
            position_utilization: Position limit utilization (0.0 to 1.0+)
            
        Returns:
            True if trigger was activated
        """
        if not self.auto_triggers_enabled:
            return False
        
        trigger_config = self.trigger_conditions[KillSwitchTrigger.POSITION_LIMIT]
        if not trigger_config["enabled"]:
            return False
        
        if position_utilization >= trigger_config["threshold"]:
            return self.trigger(
                KillSwitchTrigger.POSITION_LIMIT,
                f"Position limit exceeded: {position_utilization:.2f} >= {trigger_config['threshold']:.2f}",
                metadata={"utilization": position_utilization, "threshold": trigger_config["threshold"]},
                auto_triggered=True
            )
        
        return False
    
    def check_order_rate_trigger(self, orders_per_minute: int) -> bool:
        """
        Check if order rate trigger should activate.
        
        Args:
            orders_per_minute: Current order rate
            
        Returns:
            True if trigger was activated
        """
        if not self.auto_triggers_enabled:
            return False
        
        trigger_config = self.trigger_conditions[KillSwitchTrigger.ORDER_RATE]
        if not trigger_config["enabled"]:
            return False
        
        if orders_per_minute >= trigger_config["threshold"]:
            return self.trigger(
                KillSwitchTrigger.ORDER_RATE,
                f"Order rate exceeded threshold: {orders_per_minute} >= {trigger_config['threshold']}",
                metadata={"orders_per_minute": orders_per_minute, "threshold": trigger_config["threshold"]},
                auto_triggered=True
            )
        
        return False
    
    def check_connection_trigger(self) -> bool:
        """
        Check if connection loss trigger should activate.
        
        Returns:
            True if trigger was activated
        """
        if not self.auto_triggers_enabled:
            return False
        
        trigger_config = self.trigger_conditions[KillSwitchTrigger.CONNECTION_LOSS]
        if not trigger_config["enabled"]:
            return False
        
        time_since_connection = time.time() - self.connection_last_seen
        if time_since_connection >= trigger_config["timeout_seconds"]:
            return self.trigger(
                KillSwitchTrigger.CONNECTION_LOSS,
                f"Connection lost for {time_since_connection:.1f} seconds",
                metadata={"time_since_connection": time_since_connection, "threshold": trigger_config["timeout_seconds"]},
                auto_triggered=True
            )
        
        return False
    
    def record_connection(self) -> None:
        """Record successful connection."""
        self.connection_last_seen = time.time()
    
    def record_error(self) -> bool:
        """
        Record a system error and check if error trigger should activate.
        
        Returns:
            True if trigger was activated
        """
        current_time = time.time()
        
        # Reset error window if it's been more than 5 minutes
        if current_time - self.error_window_start > 300:
            self.error_count = 0
            self.error_window_start = current_time
        
        self.error_count += 1
        
        if not self.auto_triggers_enabled:
            return False
        
        trigger_config = self.trigger_conditions[KillSwitchTrigger.SYSTEM_ERROR]
        if not trigger_config["enabled"]:
            return False
        
        if self.error_count >= trigger_config["error_count"]:
            return self.trigger(
                KillSwitchTrigger.SYSTEM_ERROR,
                f"System error count exceeded: {self.error_count} >= {trigger_config['error_count']}",
                metadata={"error_count": self.error_count, "threshold": trigger_config["error_count"]},
                auto_triggered=True
            )
        
        return False
    
    def add_pre_shutdown_callback(self, callback: Callable[[], None]) -> None:
        """Add callback to execute before shutdown."""
        self.pre_shutdown_callbacks.append(callback)
    
    def add_shutdown_callback(self, callback: Callable[[], None]) -> None:
        """Add callback to execute during shutdown."""
        self.shutdown_callbacks.append(callback)
    
    def add_post_shutdown_callback(self, callback: Callable[[], None]) -> None:
        """Add callback to execute after shutdown."""
        self.post_shutdown_callbacks.append(callback)
    
    def enable_auto_trigger(self, trigger_type: KillSwitchTrigger) -> None:
        """Enable automatic trigger for a specific type."""
        if trigger_type in self.trigger_conditions:
            self.trigger_conditions[trigger_type]["enabled"] = True
            self.logger.info("Enabled auto trigger", trigger_type=trigger_type.value)
    
    def disable_auto_trigger(self, trigger_type: KillSwitchTrigger) -> None:
        """Disable automatic trigger for a specific type."""
        if trigger_type in self.trigger_conditions:
            self.trigger_conditions[trigger_type]["enabled"] = False
            self.logger.warning("Disabled auto trigger", trigger_type=trigger_type.value)
    
    def set_trigger_threshold(self, trigger_type: KillSwitchTrigger, threshold: float) -> None:
        """Set threshold for a specific trigger type."""
        if trigger_type in self.trigger_conditions:
            old_threshold = self.trigger_conditions[trigger_type].get("threshold")
            self.trigger_conditions[trigger_type]["threshold"] = threshold
            self.logger.info(
                "Updated trigger threshold",
                trigger_type=trigger_type.value,
                old_threshold=old_threshold,
                new_threshold=threshold
            )
    
    def is_active(self) -> bool:
        """Check if kill switch is active (not triggered)."""
        return self.status == KillSwitchStatus.ACTIVE
    
    def is_triggered(self) -> bool:
        """Check if kill switch has been triggered."""
        return self.status in [KillSwitchStatus.TRIGGERED, KillSwitchStatus.SHUTTING_DOWN, KillSwitchStatus.SHUTDOWN]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current kill switch status and configuration."""
        return {
            "enabled": self.enabled,
            "status": self.status.value,
            "auto_triggers_enabled": self.auto_triggers_enabled,
            "trigger_conditions": self.trigger_conditions,
            "last_activation": self.last_activation.reason if self.last_activation else None,
            "activation_count": len(self.activation_history),
            "connection_last_seen": self.connection_last_seen,
            "error_count": self.error_count,
            "shutdown_start_time": self.shutdown_start_time,
        }
    
    def reset(self) -> bool:
        """
        Reset kill switch to active state.
        
        This should only be used after addressing the underlying issues
        that caused the kill switch to trigger.
        
        Returns:
            True if reset was successful
        """
        if self.status == KillSwitchStatus.ACTIVE:
            self.logger.warning("Kill switch already active")
            return False
        
        # Reset state
        self.status = KillSwitchStatus.ACTIVE
        self.error_count = 0
        self.error_window_start = time.time()
        self.connection_last_seen = time.time()
        self.shutdown_start_time = None
        
        # Cancel any pending shutdown tasks
        for task in self.shutdown_tasks:
            if not task.done():
                task.cancel()
        self.shutdown_tasks.clear()
        
        self.logger.warning("Kill switch reset to active state")
        return True
    
    async def _execute_shutdown(self) -> None:
        """Execute the shutdown sequence."""
        try:
            self.status = KillSwitchStatus.SHUTTING_DOWN
            self.shutdown_start_time = time.time()
            
            self.logger.critical("Starting emergency shutdown sequence")
            
            # Phase 1: Pre-shutdown callbacks
            self.logger.info("Executing pre-shutdown callbacks")
            for callback in self.pre_shutdown_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    self.logger.error("Pre-shutdown callback error", error=str(e))
            
            # Phase 2: Main shutdown callbacks
            self.logger.info("Executing main shutdown callbacks")
            for callback in self.shutdown_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    self.logger.error("Shutdown callback error", error=str(e))
            
            # Phase 3: Post-shutdown callbacks
            self.logger.info("Executing post-shutdown callbacks")
            for callback in self.post_shutdown_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    self.logger.error("Post-shutdown callback error", error=str(e))
            
            self.status = KillSwitchStatus.SHUTDOWN
            shutdown_duration = time.time() - self.shutdown_start_time
            
            self.logger.critical(
                "Emergency shutdown completed",
                duration_seconds=shutdown_duration
            )
            
        except Exception as e:
            self.logger.error("Error during shutdown sequence", error=str(e))
            self.status = KillSwitchStatus.SHUTDOWN


# Global kill switch instance
_global_kill_switch: Optional[KillSwitch] = None


def get_global_kill_switch() -> KillSwitch:
    """Get the global kill switch instance."""
    global _global_kill_switch
    if _global_kill_switch is None:
        _global_kill_switch = KillSwitch()
    return _global_kill_switch


def set_global_kill_switch(kill_switch: KillSwitch) -> None:
    """Set the global kill switch instance."""
    global _global_kill_switch
    _global_kill_switch = kill_switch


def emergency_stop(reason: str = "Emergency stop") -> bool:
    """Trigger global emergency stop."""
    kill_switch = get_global_kill_switch()
    return kill_switch.manual_trigger(reason)
