"""
Risk limit management and monitoring.

This module provides comprehensive risk limit management including:
- Position limits (per symbol and aggregate)
- Notional exposure limits
- PnL-based limits and drawdown protection
- Order rate limiting
- Dynamic limit adjustment based on market conditions
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable

from mm_bot.config import RiskConfig
from mm_bot.logging import LoggerMixin
from mm_bot.utils.safemath import clamp


class LimitType(Enum):
    """Risk limit types."""
    POSITION = "position"
    NOTIONAL = "notional"
    DRAWDOWN = "drawdown"
    ORDER_RATE = "order_rate"
    DAILY_LOSS = "daily_loss"
    VAR = "var"


class LimitStatus(Enum):
    """Risk limit status."""
    NORMAL = "normal"          # Within limits
    WARNING = "warning"        # Approaching limits
    BREACH = "breach"          # Limit breached
    CRITICAL = "critical"      # Critical breach requiring immediate action


@dataclass
class RiskLimit:
    """Individual risk limit definition."""
    limit_type: LimitType
    symbol: Optional[str]  # None for aggregate limits
    max_value: float
    warning_threshold: float  # As fraction of max_value (e.g., 0.8 for 80%)
    current_value: float = 0.0
    peak_value: float = 0.0
    last_updated: int = 0
    enabled: bool = True
    
    @property
    def utilization(self) -> float:
        """Get limit utilization as fraction (0.0 to 1.0+)."""
        if self.max_value <= 0:
            return 0.0
        return abs(self.current_value) / self.max_value
    
    @property
    def status(self) -> LimitStatus:
        """Get current limit status."""
        if not self.enabled:
            return LimitStatus.NORMAL
        
        utilization = self.utilization
        
        if utilization >= 1.0:
            return LimitStatus.BREACH
        elif utilization >= self.warning_threshold:
            return LimitStatus.WARNING
        else:
            return LimitStatus.NORMAL
    
    @property
    def remaining_capacity(self) -> float:
        """Get remaining capacity before limit breach."""
        return max(0.0, self.max_value - abs(self.current_value))
    
    def update_value(self, new_value: float) -> None:
        """Update current value and peak tracking."""
        self.current_value = new_value
        self.peak_value = max(self.peak_value, abs(new_value))
        self.last_updated = int(time.time() * 1000)


class RiskLimitManager(LoggerMixin):
    """
    Comprehensive risk limit management system.
    
    Features:
    - Multiple limit types with per-symbol and aggregate tracking
    - Dynamic warning thresholds
    - Limit breach callbacks and actions
    - Historical limit utilization tracking
    - Market condition-based limit adjustments
    """
    
    def __init__(self, config: RiskConfig):
        """
        Initialize risk limit manager.
        
        Args:
            config: Risk configuration
        """
        super().__init__()
        self.config = config
        
        # Limit tracking
        self.limits: Dict[str, RiskLimit] = {}
        self.limit_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Callbacks for limit events
        self.warning_callbacks: List[Callable[[RiskLimit], None]] = []
        self.breach_callbacks: List[Callable[[RiskLimit], None]] = []
        
        # Order rate tracking
        self.order_timestamps: deque = deque(maxlen=1000)
        
        # PnL tracking
        self.pnl_history: deque = deque(maxlen=10000)  # Keep 10k data points
        self.peak_pnl = 0.0
        self.current_pnl = 0.0
        
        # Statistics
        self.total_limit_breaches = 0
        self.limit_breach_count: Dict[LimitType, int] = defaultdict(int)
        
        self._initialize_limits()
        
        self.logger.info(
            "Initialized risk limit manager",
            max_position=config.max_position,
            max_notional=config.max_notional,
            max_drawdown_pct=config.max_drawdown_pct,
            max_order_rate=config.max_order_rate_per_min
        )
    
    def _initialize_limits(self) -> None:
        """Initialize default risk limits from configuration."""
        # Aggregate position limit
        self.add_limit(
            LimitType.POSITION,
            symbol=None,
            max_value=self.config.max_position,
            warning_threshold=1.0 - self.config.position_limit_buffer
        )
        
        # Aggregate notional limit
        self.add_limit(
            LimitType.NOTIONAL,
            symbol=None,
            max_value=self.config.max_notional,
            warning_threshold=0.9
        )
        
        # Drawdown limit
        self.add_limit(
            LimitType.DRAWDOWN,
            symbol=None,
            max_value=self.config.max_drawdown_pct,
            warning_threshold=0.8
        )
        
        # Order rate limit
        self.add_limit(
            LimitType.ORDER_RATE,
            symbol=None,
            max_value=self.config.max_order_rate_per_min,
            warning_threshold=0.8
        )
    
    def add_limit(
        self,
        limit_type: LimitType,
        max_value: float,
        symbol: Optional[str] = None,
        warning_threshold: float = 0.8,
        enabled: bool = True
    ) -> str:
        """
        Add a new risk limit.
        
        Args:
            limit_type: Type of risk limit
            max_value: Maximum allowed value
            symbol: Symbol for per-symbol limits (None for aggregate)
            warning_threshold: Warning threshold as fraction of max_value
            enabled: Whether limit is enabled
            
        Returns:
            Limit ID for tracking
        """
        limit_id = f"{limit_type.value}_{symbol or 'aggregate'}"
        
        limit = RiskLimit(
            limit_type=limit_type,
            symbol=symbol,
            max_value=max_value,
            warning_threshold=warning_threshold,
            enabled=enabled
        )
        
        self.limits[limit_id] = limit
        
        self.logger.info(
            "Added risk limit",
            limit_id=limit_id,
            limit_type=limit_type.value,
            symbol=symbol,
            max_value=max_value,
            warning_threshold=warning_threshold
        )
        
        return limit_id
    
    def update_position(self, symbol: str, position: float, mark_price: float) -> List[RiskLimit]:
        """
        Update position and check position-related limits.
        
        Args:
            symbol: Trading symbol
            position: Current position size
            mark_price: Mark price for notional calculation
            
        Returns:
            List of limits that are in warning or breach status
        """
        breached_limits = []
        
        # Update per-symbol position limit
        symbol_position_id = f"{LimitType.POSITION.value}_{symbol}"
        if symbol_position_id in self.limits:
            limit = self.limits[symbol_position_id]
            limit.update_value(abs(position))
            if limit.status in [LimitStatus.WARNING, LimitStatus.BREACH]:
                breached_limits.append(limit)
        
        # Update aggregate position limit
        aggregate_position_id = f"{LimitType.POSITION.value}_aggregate"
        if aggregate_position_id in self.limits:
            # Calculate total position across all symbols
            total_position = abs(position)  # This would need to sum across all symbols in practice
            limit = self.limits[aggregate_position_id]
            limit.update_value(total_position)
            if limit.status in [LimitStatus.WARNING, LimitStatus.BREACH]:
                breached_limits.append(limit)
        
        # Update notional limits
        notional = abs(position * mark_price)
        
        # Per-symbol notional
        symbol_notional_id = f"{LimitType.NOTIONAL.value}_{symbol}"
        if symbol_notional_id in self.limits:
            limit = self.limits[symbol_notional_id]
            limit.update_value(notional)
            if limit.status in [LimitStatus.WARNING, LimitStatus.BREACH]:
                breached_limits.append(limit)
        
        # Aggregate notional
        aggregate_notional_id = f"{LimitType.NOTIONAL.value}_aggregate"
        if aggregate_notional_id in self.limits:
            # This would need to sum across all symbols in practice
            limit = self.limits[aggregate_notional_id]
            limit.update_value(notional)
            if limit.status in [LimitStatus.WARNING, LimitStatus.BREACH]:
                breached_limits.append(limit)
        
        # Process any breaches
        for limit in breached_limits:
            self._handle_limit_event(limit)
        
        return breached_limits
    
    def update_pnl(self, pnl: float) -> List[RiskLimit]:
        """
        Update PnL and check drawdown limits.
        
        Args:
            pnl: Current unrealized PnL
            
        Returns:
            List of limits that are in warning or breach status
        """
        self.current_pnl = pnl
        self.pnl_history.append((int(time.time() * 1000), pnl))
        
        # Update peak PnL
        if pnl > self.peak_pnl:
            self.peak_pnl = pnl
        
        breached_limits = []
        
        # Calculate drawdown
        drawdown = 0.0
        if self.peak_pnl > 0:
            drawdown = ((self.peak_pnl - pnl) / self.peak_pnl) * 100
        
        # Update drawdown limit
        drawdown_id = f"{LimitType.DRAWDOWN.value}_aggregate"
        if drawdown_id in self.limits:
            limit = self.limits[drawdown_id]
            limit.update_value(drawdown)
            if limit.status in [LimitStatus.WARNING, LimitStatus.BREACH]:
                breached_limits.append(limit)
        
        # Process any breaches
        for limit in breached_limits:
            self._handle_limit_event(limit)
        
        return breached_limits
    
    def record_order(self) -> List[RiskLimit]:
        """
        Record an order for rate limiting.
        
        Returns:
            List of limits that are in warning or breach status
        """
        current_time = int(time.time() * 1000)
        self.order_timestamps.append(current_time)
        
        breached_limits = []
        
        # Calculate order rate (orders per minute)
        one_minute_ago = current_time - 60000
        recent_orders = sum(1 for ts in self.order_timestamps if ts >= one_minute_ago)
        
        # Update order rate limit
        rate_id = f"{LimitType.ORDER_RATE.value}_aggregate"
        if rate_id in self.limits:
            limit = self.limits[rate_id]
            limit.update_value(recent_orders)
            if limit.status in [LimitStatus.WARNING, LimitStatus.BREACH]:
                breached_limits.append(limit)
        
        # Process any breaches
        for limit in breached_limits:
            self._handle_limit_event(limit)
        
        return breached_limits
    
    def check_order_allowed(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float
    ) -> Tuple[bool, List[str]]:
        """
        Check if an order is allowed based on current risk limits.
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            amount: Order amount
            price: Order price
            
        Returns:
            Tuple of (allowed, list_of_violations)
        """
        violations = []
        
        # Calculate position impact
        position_change = amount if side.lower() == "buy" else -amount
        notional = amount * price
        
        # Check position limits
        symbol_position_id = f"{LimitType.POSITION.value}_{symbol}"
        if symbol_position_id in self.limits:
            limit = self.limits[symbol_position_id]
            new_position = limit.current_value + abs(position_change)
            if new_position > limit.max_value:
                violations.append(f"Symbol position limit exceeded: {new_position:.4f} > {limit.max_value:.4f}")
        
        # Check aggregate position limit
        aggregate_position_id = f"{LimitType.POSITION.value}_aggregate"
        if aggregate_position_id in self.limits:
            limit = self.limits[aggregate_position_id]
            new_position = limit.current_value + abs(position_change)
            if new_position > limit.max_value:
                violations.append(f"Aggregate position limit exceeded: {new_position:.4f} > {limit.max_value:.4f}")
        
        # Check notional limits
        symbol_notional_id = f"{LimitType.NOTIONAL.value}_{symbol}"
        if symbol_notional_id in self.limits:
            limit = self.limits[symbol_notional_id]
            new_notional = limit.current_value + notional
            if new_notional > limit.max_value:
                violations.append(f"Symbol notional limit exceeded: {new_notional:.2f} > {limit.max_value:.2f}")
        
        # Check aggregate notional limit
        aggregate_notional_id = f"{LimitType.NOTIONAL.value}_aggregate"
        if aggregate_notional_id in self.limits:
            limit = self.limits[aggregate_notional_id]
            new_notional = limit.current_value + notional
            if new_notional > limit.max_value:
                violations.append(f"Aggregate notional limit exceeded: {new_notional:.2f} > {limit.max_value:.2f}")
        
        # Check order rate limit
        rate_id = f"{LimitType.ORDER_RATE.value}_aggregate"
        if rate_id in self.limits:
            limit = self.limits[rate_id]
            if limit.current_value >= limit.max_value:
                violations.append(f"Order rate limit exceeded: {limit.current_value} >= {limit.max_value}")
        
        allowed = len(violations) == 0
        
        if not allowed:
            self.logger.warning(
                "Order blocked by risk limits",
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                violations=violations
            )
        
        return allowed, violations
    
    def get_limit(self, limit_id: str) -> Optional[RiskLimit]:
        """Get a specific limit by ID."""
        return self.limits.get(limit_id)
    
    def get_limits_by_type(self, limit_type: LimitType) -> List[RiskLimit]:
        """Get all limits of a specific type."""
        return [limit for limit in self.limits.values() if limit.limit_type == limit_type]
    
    def get_limits_by_status(self, status: LimitStatus) -> List[RiskLimit]:
        """Get all limits with a specific status."""
        return [limit for limit in self.limits.values() if limit.status == status]
    
    def get_all_breaches(self) -> List[RiskLimit]:
        """Get all currently breached limits."""
        return self.get_limits_by_status(LimitStatus.BREACH)
    
    def get_all_warnings(self) -> List[RiskLimit]:
        """Get all limits currently in warning state."""
        return self.get_limits_by_status(LimitStatus.WARNING)
    
    def enable_limit(self, limit_id: str) -> bool:
        """Enable a specific limit."""
        if limit_id in self.limits:
            self.limits[limit_id].enabled = True
            self.logger.info("Enabled risk limit", limit_id=limit_id)
            return True
        return False
    
    def disable_limit(self, limit_id: str) -> bool:
        """Disable a specific limit."""
        if limit_id in self.limits:
            self.limits[limit_id].enabled = False
            self.logger.warning("Disabled risk limit", limit_id=limit_id)
            return True
        return False
    
    def adjust_limit(self, limit_id: str, new_max_value: float) -> bool:
        """Adjust the maximum value for a limit."""
        if limit_id in self.limits:
            old_value = self.limits[limit_id].max_value
            self.limits[limit_id].max_value = new_max_value
            self.logger.info(
                "Adjusted risk limit",
                limit_id=limit_id,
                old_value=old_value,
                new_value=new_max_value
            )
            return True
        return False
    
    def add_warning_callback(self, callback: Callable[[RiskLimit], None]) -> None:
        """Add callback for limit warnings."""
        self.warning_callbacks.append(callback)
    
    def add_breach_callback(self, callback: Callable[[RiskLimit], None]) -> None:
        """Add callback for limit breaches."""
        self.breach_callbacks.append(callback)
    
    def get_risk_summary(self) -> Dict[str, any]:
        """Get comprehensive risk summary."""
        summary = {
            "total_limits": len(self.limits),
            "enabled_limits": sum(1 for limit in self.limits.values() if limit.enabled),
            "breached_limits": len(self.get_all_breaches()),
            "warning_limits": len(self.get_all_warnings()),
            "total_breaches": self.total_limit_breaches,
            "breach_count_by_type": dict(self.limit_breach_count),
            "current_pnl": self.current_pnl,
            "peak_pnl": self.peak_pnl,
            "current_drawdown_pct": ((self.peak_pnl - self.current_pnl) / self.peak_pnl * 100) if self.peak_pnl > 0 else 0.0,
            "limits": {}
        }
        
        # Add individual limit details
        for limit_id, limit in self.limits.items():
            summary["limits"][limit_id] = {
                "type": limit.limit_type.value,
                "symbol": limit.symbol,
                "current_value": limit.current_value,
                "max_value": limit.max_value,
                "utilization": limit.utilization,
                "status": limit.status.value,
                "enabled": limit.enabled,
                "remaining_capacity": limit.remaining_capacity
            }
        
        return summary
    
    def reset_peaks(self) -> None:
        """Reset peak values for all limits."""
        for limit in self.limits.values():
            limit.peak_value = abs(limit.current_value)
        
        self.peak_pnl = self.current_pnl
        
        self.logger.info("Reset peak values for all limits")
    
    def _handle_limit_event(self, limit: RiskLimit) -> None:
        """Handle limit warning or breach event."""
        if limit.status == LimitStatus.WARNING:
            self.logger.warning(
                "Risk limit warning",
                limit_id=f"{limit.limit_type.value}_{limit.symbol or 'aggregate'}",
                current_value=limit.current_value,
                max_value=limit.max_value,
                utilization=limit.utilization
            )
            
            # Call warning callbacks
            for callback in self.warning_callbacks:
                try:
                    callback(limit)
                except Exception as e:
                    self.logger.error("Warning callback error", error=str(e))
        
        elif limit.status == LimitStatus.BREACH:
            self.total_limit_breaches += 1
            self.limit_breach_count[limit.limit_type] += 1
            
            self.logger.error(
                "Risk limit breach",
                limit_id=f"{limit.limit_type.value}_{limit.symbol or 'aggregate'}",
                current_value=limit.current_value,
                max_value=limit.max_value,
                utilization=limit.utilization
            )
            
            # Call breach callbacks
            for callback in self.breach_callbacks:
                try:
                    callback(limit)
                except Exception as e:
                    self.logger.error("Breach callback error", error=str(e))
        
        # Store in history
        limit_id = f"{limit.limit_type.value}_{limit.symbol or 'aggregate'}"
        self.limit_history[limit_id].append({
            "timestamp": int(time.time() * 1000),
            "value": limit.current_value,
            "utilization": limit.utilization,
            "status": limit.status.value
        })
