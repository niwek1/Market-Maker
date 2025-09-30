"""
Inventory management and position tracking.

This module provides comprehensive inventory management including:
- Real-time position tracking
- Inventory target management
- Mean reversion calculations
- Risk-adjusted inventory limits
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Optional, List
import time

from mm_bot.logging import LoggerMixin
from mm_bot.utils.safemath import clamp


@dataclass
class InventorySnapshot:
    """Snapshot of inventory at a point in time."""
    timestamp: int
    symbol: str
    position: float
    notional: float
    mark_price: float
    target: float
    deviation: float


class InventoryManager(LoggerMixin):
    """
    Comprehensive inventory management system.
    
    Features:
    - Real-time position tracking
    - Inventory target management
    - Mean reversion monitoring
    - Risk-adjusted position limits
    - Historical inventory analysis
    """
    
    def __init__(self, inventory_target: float = 0.0, inventory_band: float = 0.1):
        """
        Initialize inventory manager.
        
        Args:
            inventory_target: Target inventory level
            inventory_band: Acceptable deviation from target
        """
        super().__init__()
        self.inventory_target = inventory_target
        self.inventory_band = inventory_band
        
        # Position tracking
        self.positions: Dict[str, float] = defaultdict(float)
        self.mark_prices: Dict[str, float] = {}
        
        # Historical tracking
        self.inventory_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Statistics
        self.total_trades = 0
        self.inventory_changes = 0
        
        self.logger.info(
            "Initialized inventory manager",
            target=inventory_target,
            band=inventory_band
        )
    
    def update_position(self, symbol: str, position: float, mark_price: float) -> None:
        """
        Update position for a symbol.
        
        Args:
            symbol: Trading symbol
            position: New position size
            mark_price: Current mark price
        """
        old_position = self.positions[symbol]
        self.positions[symbol] = position
        self.mark_prices[symbol] = mark_price
        
        # Track inventory change
        if abs(position - old_position) > 1e-8:
            self.inventory_changes += 1
        
        # Create snapshot
        snapshot = InventorySnapshot(
            timestamp=int(time.time() * 1000),
            symbol=symbol,
            position=position,
            notional=position * mark_price,
            mark_price=mark_price,
            target=self.inventory_target,
            deviation=position - self.inventory_target
        )
        
        # Store in history
        self.inventory_history[symbol].append(snapshot)
        
        self.logger.debug(
            "Updated position",
            symbol=symbol,
            old_position=old_position,
            new_position=position,
            mark_price=mark_price,
            notional=snapshot.notional
        )
    
    def get_position(self, symbol: str) -> float:
        """Get current position for a symbol."""
        return self.positions.get(symbol, 0.0)
    
    def get_notional(self, symbol: str) -> float:
        """Get current notional position for a symbol."""
        position = self.positions.get(symbol, 0.0)
        mark_price = self.mark_prices.get(symbol, 0.0)
        return position * mark_price
    
    def get_total_notional(self) -> float:
        """Get total notional across all positions."""
        return sum(
            pos * self.mark_prices.get(symbol, 0.0)
            for symbol, pos in self.positions.items()
        )
    
    def get_inventory_deviation(self, symbol: str) -> float:
        """Get deviation from inventory target."""
        position = self.positions.get(symbol, 0.0)
        return position - self.inventory_target
    
    def is_within_band(self, symbol: str) -> bool:
        """Check if position is within inventory band."""
        deviation = abs(self.get_inventory_deviation(symbol))
        return deviation <= self.inventory_band
    
    def get_mean_reversion_signal(self, symbol: str) -> float:
        """
        Get mean reversion signal strength.
        
        Returns:
            Signal from -1.0 to 1.0, where negative means sell pressure
            and positive means buy pressure to revert to target.
        """
        deviation = self.get_inventory_deviation(symbol)
        
        # Normalize by band size
        if self.inventory_band > 0:
            normalized_deviation = deviation / self.inventory_band
        else:
            normalized_deviation = deviation
        
        # Return opposite signal (negative deviation = buy signal)
        return clamp(-normalized_deviation, -1.0, 1.0)
    
    def get_inventory_risk(self, symbol: str, max_position: float) -> float:
        """
        Get inventory risk level (0.0 to 1.0+).
        
        Args:
            symbol: Trading symbol
            max_position: Maximum allowed position
            
        Returns:
            Risk level where 1.0 = at limit, >1.0 = over limit
        """
        if max_position <= 0:
            return 0.0
        
        position = abs(self.positions.get(symbol, 0.0))
        return position / max_position
    
    def calculate_optimal_inventory_target(
        self,
        symbol: str,
        volatility: float,
        time_horizon: float = 1.0
    ) -> float:
        """
        Calculate optimal inventory target based on market conditions.
        
        Args:
            symbol: Trading symbol
            volatility: Current volatility
            time_horizon: Time horizon in days
            
        Returns:
            Optimal inventory target
        """
        # In high volatility, target should be closer to zero
        # In low volatility, can hold more inventory
        vol_adjustment = max(0.1, min(2.0, 0.02 / max(volatility, 0.001)))
        
        # Get historical inventory performance
        history = list(self.inventory_history[symbol])
        if len(history) < 10:
            return self.inventory_target
        
        # Simple momentum-based adjustment
        recent_positions = [snap.position for snap in history[-10:]]
        if recent_positions:
            momentum = (recent_positions[-1] - recent_positions[0]) / len(recent_positions)
            momentum_adjustment = clamp(momentum * 0.1, -0.05, 0.05)
        else:
            momentum_adjustment = 0.0
        
        optimal_target = self.inventory_target * vol_adjustment + momentum_adjustment
        
        self.logger.debug(
            "Calculated optimal inventory target",
            symbol=symbol,
            base_target=self.inventory_target,
            volatility=volatility,
            vol_adjustment=vol_adjustment,
            momentum_adjustment=momentum_adjustment,
            optimal_target=optimal_target
        )
        
        return optimal_target
    
    def get_inventory_statistics(self, symbol: str) -> Dict[str, any]:
        """Get comprehensive inventory statistics."""
        history = list(self.inventory_history[symbol])
        if not history:
            return {}
        
        positions = [snap.position for snap in history]
        deviations = [snap.deviation for snap in history]
        notionals = [snap.notional for snap in history]
        
        import numpy as np
        
        stats = {
            "current_position": self.positions.get(symbol, 0.0),
            "current_notional": self.get_notional(symbol),
            "target": self.inventory_target,
            "current_deviation": self.get_inventory_deviation(symbol),
            "within_band": self.is_within_band(symbol),
            "mean_reversion_signal": self.get_mean_reversion_signal(symbol),
            "history_length": len(history),
        }
        
        if len(positions) > 1:
            stats.update({
                "position_mean": np.mean(positions),
                "position_std": np.std(positions),
                "position_min": np.min(positions),
                "position_max": np.max(positions),
                "deviation_mean": np.mean(deviations),
                "deviation_std": np.std(deviations),
                "notional_mean": np.mean(notionals),
                "notional_std": np.std(notionals),
                "time_in_band_pct": sum(1 for d in deviations if abs(d) <= self.inventory_band) / len(deviations) * 100,
            })
        
        return stats
    
    def set_inventory_target(self, new_target: float, reason: str = "manual") -> None:
        """Update inventory target."""
        old_target = self.inventory_target
        self.inventory_target = new_target
        
        self.logger.info(
            "Updated inventory target",
            old_target=old_target,
            new_target=new_target,
            reason=reason
        )
    
    def set_inventory_band(self, new_band: float) -> None:
        """Update inventory band."""
        old_band = self.inventory_band
        self.inventory_band = new_band
        
        self.logger.info(
            "Updated inventory band",
            old_band=old_band,
            new_band=new_band
        )
    
    def reset_position(self, symbol: str, reason: str = "reset") -> None:
        """Reset position for a symbol."""
        old_position = self.positions.get(symbol, 0.0)
        self.positions[symbol] = 0.0
        
        self.logger.warning(
            "Reset position",
            symbol=symbol,
            old_position=old_position,
            reason=reason
        )
    
    def get_all_positions(self) -> Dict[str, float]:
        """Get all current positions."""
        return dict(self.positions)
    
    def get_summary(self) -> Dict[str, any]:
        """Get inventory manager summary."""
        return {
            "inventory_target": self.inventory_target,
            "inventory_band": self.inventory_band,
            "total_symbols": len(self.positions),
            "total_notional": self.get_total_notional(),
            "total_trades": self.total_trades,
            "inventory_changes": self.inventory_changes,
            "positions": dict(self.positions),
            "mark_prices": dict(self.mark_prices),
        }
