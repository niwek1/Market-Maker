"""
Hedging module for delta-neutral position management.

This module provides hedging capabilities to maintain delta neutrality
by offsetting positions across different instruments or exchanges.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from mm_bot.config import HedgeConfig
from mm_bot.logging import LoggerMixin


class HedgeDirection(Enum):
    """Hedge direction."""
    LONG = "long"
    SHORT = "short"


@dataclass
class HedgeOrder:
    """Hedge order specification."""
    symbol: str
    direction: HedgeDirection
    size: float
    target_price: Optional[float] = None
    urgency: float = 0.5  # 0.0 = passive, 1.0 = aggressive


class Hedger(LoggerMixin):
    """
    Delta hedging implementation for position management.
    
    Features:
    - Cross-instrument hedging (e.g., spot vs futures)
    - Cross-exchange hedging
    - Dynamic hedge ratio calculation
    - Risk-aware hedge sizing
    """
    
    def __init__(self, config: HedgeConfig):
        """
        Initialize hedger.
        
        Args:
            config: Hedge configuration
        """
        super().__init__()
        self.config = config
        self.enabled = config.enabled
        
        # Hedge tracking
        self.hedge_positions: Dict[str, float] = {}
        self.hedge_orders: List[HedgeOrder] = []
        
        self.logger.info(
            "Initialized hedger",
            enabled=self.enabled,
            hedge_symbol=config.hedge_symbol,
            threshold=config.hedge_threshold
        )
    
    def calculate_hedge_requirement(
        self,
        primary_symbol: str,
        primary_position: float,
        primary_price: float
    ) -> Optional[HedgeOrder]:
        """
        Calculate hedge requirement for a primary position.
        
        Args:
            primary_symbol: Primary trading symbol
            primary_position: Current position in primary symbol
            primary_price: Current price of primary symbol
            
        Returns:
            HedgeOrder if hedging is required, None otherwise
        """
        if not self.enabled or not self.config.hedge_symbol:
            return None
        
        # Calculate current hedge position
        current_hedge = self.hedge_positions.get(self.config.hedge_symbol, 0.0)
        
        # Calculate target hedge position
        target_hedge = -primary_position * self.config.hedge_size_ratio
        
        # Calculate hedge delta
        hedge_delta = target_hedge - current_hedge
        
        # Check if hedge is needed
        if abs(hedge_delta) < self.config.hedge_threshold:
            return None
        
        # Determine hedge direction
        direction = HedgeDirection.LONG if hedge_delta > 0 else HedgeDirection.SHORT
        
        hedge_order = HedgeOrder(
            symbol=self.config.hedge_symbol,
            direction=direction,
            size=abs(hedge_delta),
            urgency=min(1.0, abs(hedge_delta) / self.config.hedge_threshold)
        )
        
        self.logger.info(
            "Hedge required",
            primary_symbol=primary_symbol,
            primary_position=primary_position,
            hedge_symbol=self.config.hedge_symbol,
            current_hedge=current_hedge,
            target_hedge=target_hedge,
            hedge_delta=hedge_delta,
            direction=direction.value,
            size=hedge_order.size
        )
        
        return hedge_order
    
    def update_hedge_position(self, symbol: str, position: float) -> None:
        """Update hedge position tracking."""
        self.hedge_positions[symbol] = position
        
        self.logger.debug(
            "Updated hedge position",
            symbol=symbol,
            position=position
        )
    
    def get_hedge_status(self) -> Dict[str, any]:
        """Get current hedge status."""
        return {
            "enabled": self.enabled,
            "hedge_symbol": self.config.hedge_symbol,
            "hedge_threshold": self.config.hedge_threshold,
            "hedge_size_ratio": self.config.hedge_size_ratio,
            "hedge_positions": dict(self.hedge_positions),
            "pending_orders": len(self.hedge_orders)
        }
