"""
Level 2 order book cache with microstructure feature calculations.

This module provides real-time order book management with calculations for:
- Microprice (volume-weighted mid price)
- Order book imbalance (OBI)
- Short-term volatility
- Spread analysis
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np
import pandas as pd

from mm_bot.logging import LoggerMixin
from mm_bot.utils.safemath import safe_divide, safe_log, safe_sqrt
from mm_bot.utils.time import timestamp_ms


class OrderBookSide(Enum):
    """Order book side enumeration."""
    BID = "bid"
    ASK = "ask"


@dataclass
class OrderBookLevel:
    """Single order book level."""
    price: float
    size: float
    timestamp: int  # milliseconds
    
    @property
    def notional(self) -> float:
        """Get notional value of this level."""
        return self.price * self.size


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot."""
    symbol: str
    timestamp: int
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    
    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        """Get best bid level."""
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        """Get best ask level."""
        return self.asks[0] if self.asks else None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Get mid price."""
        if not self.bids or not self.asks:
            return None
        return (self.bids[0].price + self.asks[0].price) / 2.0
    
    @property
    def spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        if not self.bids or not self.asks:
            return None
        return self.asks[0].price - self.bids[0].price
    
    @property
    def spread_bps(self) -> Optional[float]:
        """Get spread in basis points."""
        mid = self.mid_price
        spread = self.spread
        if mid is None or spread is None or mid <= 0:
            return None
        return (spread / mid) * 10000


@dataclass
class MicrostructureFeatures:
    """Microstructure features calculated from order book."""
    timestamp: int
    symbol: str
    
    # Basic prices
    bid: float
    ask: float
    mid: float
    microprice: float
    
    # Spread metrics
    spread: float
    spread_bps: float
    
    # Volume metrics
    bid_size: float
    ask_size: float
    bid_notional: float
    ask_notional: float
    
    # Imbalance metrics
    volume_imbalance: float  # (bid_vol - ask_vol) / (bid_vol + ask_vol)
    notional_imbalance: float  # (bid_notional - ask_notional) / (bid_notional + ask_notional)
    
    # Depth metrics
    depth_imbalance: float  # Imbalance considering multiple levels
    weighted_mid: float  # Size-weighted mid price
    
    # Volatility
    short_vol: Optional[float] = None  # Short-term volatility
    
    def to_dict(self) -> Dict[str, Union[float, int, str, None]]:
        """Convert to dictionary for logging/metrics."""
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "bid": self.bid,
            "ask": self.ask,
            "mid": self.mid,
            "microprice": self.microprice,
            "spread": self.spread,
            "spread_bps": self.spread_bps,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "volume_imbalance": self.volume_imbalance,
            "notional_imbalance": self.notional_imbalance,
            "depth_imbalance": self.depth_imbalance,
            "weighted_mid": self.weighted_mid,
            "short_vol": self.short_vol,
        }


class L2Cache(LoggerMixin):
    """
    Level 2 order book cache with microstructure feature calculations.
    
    Maintains real-time order book state and calculates various microstructure
    indicators used for market making decisions.
    """
    
    def __init__(
        self,
        symbol: str,
        max_levels: int = 20,
        vol_window_secs: int = 300,
        imbalance_window_ms: int = 1000,
        feature_history_size: int = 1000
    ):
        """
        Initialize L2 cache.
        
        Args:
            symbol: Trading symbol
            max_levels: Maximum order book levels to track
            vol_window_secs: Volatility calculation window in seconds
            imbalance_window_ms: Imbalance calculation window in milliseconds
            feature_history_size: Maximum number of feature snapshots to keep
        """
        super().__init__()
        
        self.symbol = symbol
        self.max_levels = max_levels
        self.vol_window_secs = vol_window_secs
        self.imbalance_window_ms = imbalance_window_ms
        
        # Current order book state
        self.bids: List[OrderBookLevel] = []
        self.asks: List[OrderBookLevel] = []
        self.last_update_time = 0
        
        # Feature history
        self.features_history: deque = deque(maxlen=feature_history_size)
        self.mid_prices: deque = deque(maxlen=int(vol_window_secs * 10))  # Assume ~10 updates/sec
        self.timestamps: deque = deque(maxlen=int(vol_window_secs * 10))
        
        # Statistics
        self.update_count = 0
        self.last_features: Optional[MicrostructureFeatures] = None
    
    def update_order_book(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        timestamp: Optional[int] = None
    ) -> bool:
        """
        Update order book with new data.
        
        Args:
            bids: List of (price, size) tuples for bids
            asks: List of (price, size) tuples for asks
            timestamp: Update timestamp in milliseconds
            
        Returns:
            True if update was successful
        """
        if timestamp is None:
            timestamp = timestamp_ms()
        
        try:
            # Convert to OrderBookLevel objects
            self.bids = [
                OrderBookLevel(price=price, size=size, timestamp=timestamp)
                for price, size in sorted(bids, key=lambda x: x[0], reverse=True)[:self.max_levels]
                if size > 0
            ]
            
            self.asks = [
                OrderBookLevel(price=price, size=size, timestamp=timestamp)
                for price, size in sorted(asks, key=lambda x: x[0])[:self.max_levels]
                if size > 0
            ]
            
            # Validate order book
            if not self._validate_order_book():
                self.logger.warning("Invalid order book received", symbol=self.symbol)
                return False
            
            self.last_update_time = timestamp
            self.update_count += 1
            
            # Calculate and store features
            features = self._calculate_features(timestamp)
            if features:
                self.features_history.append(features)
                self.last_features = features
                
                # Update price history for volatility
                self.mid_prices.append(features.mid)
                self.timestamps.append(timestamp)
            
            return True
            
        except Exception as e:
            self.logger.error("Error updating order book", error=str(e), symbol=self.symbol)
            return False
    
    def _validate_order_book(self) -> bool:
        """Validate order book consistency."""
        if not self.bids or not self.asks:
            return False
        
        # Check bid ordering (descending)
        for i in range(len(self.bids) - 1):
            if self.bids[i].price <= self.bids[i + 1].price:
                return False
        
        # Check ask ordering (ascending)
        for i in range(len(self.asks) - 1):
            if self.asks[i].price >= self.asks[i + 1].price:
                return False
        
        # Check no crossed book
        if self.bids[0].price >= self.asks[0].price:
            return False
        
        return True
    
    def _calculate_features(self, timestamp: int) -> Optional[MicrostructureFeatures]:
        """Calculate microstructure features from current order book."""
        if not self.bids or not self.asks:
            return None
        
        try:
            best_bid = self.bids[0]
            best_ask = self.asks[0]
            
            # Basic prices
            bid_price = best_bid.price
            ask_price = best_ask.price
            mid_price = (bid_price + ask_price) / 2.0
            
            # Spread
            spread = ask_price - bid_price
            spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 0
            
            # Volume metrics
            bid_size = best_bid.size
            ask_size = best_ask.size
            bid_notional = bid_price * bid_size
            ask_notional = ask_price * ask_size
            
            # Calculate microprice (volume-weighted mid)
            total_size = bid_size + ask_size
            if total_size > 0:
                microprice = (bid_price * ask_size + ask_price * bid_size) / total_size
            else:
                microprice = mid_price
            
            # Volume imbalance
            volume_imbalance = safe_divide(
                bid_size - ask_size,
                bid_size + ask_size,
                default=0.0
            ) or 0.0
            
            # Notional imbalance
            notional_imbalance = safe_divide(
                bid_notional - ask_notional,
                bid_notional + ask_notional,
                default=0.0
            ) or 0.0
            
            # Depth-based calculations
            depth_imbalance = self._calculate_depth_imbalance()
            weighted_mid = self._calculate_weighted_mid()
            
            # Volatility
            short_vol = self._calculate_short_volatility()
            
            return MicrostructureFeatures(
                timestamp=timestamp,
                symbol=self.symbol,
                bid=bid_price,
                ask=ask_price,
                mid=mid_price,
                microprice=microprice,
                spread=spread,
                spread_bps=spread_bps,
                bid_size=bid_size,
                ask_size=ask_size,
                bid_notional=bid_notional,
                ask_notional=ask_notional,
                volume_imbalance=volume_imbalance,
                notional_imbalance=notional_imbalance,
                depth_imbalance=depth_imbalance,
                weighted_mid=weighted_mid,
                short_vol=short_vol,
            )
            
        except Exception as e:
            self.logger.error("Error calculating features", error=str(e), symbol=self.symbol)
            return None
    
    def _calculate_depth_imbalance(self, levels: int = 5) -> float:
        """Calculate depth imbalance using multiple levels."""
        try:
            max_levels = min(levels, len(self.bids), len(self.asks))
            if max_levels == 0:
                return 0.0
            
            bid_depth = sum(level.size for level in self.bids[:max_levels])
            ask_depth = sum(level.size for level in self.asks[:max_levels])
            
            return safe_divide(
                bid_depth - ask_depth,
                bid_depth + ask_depth,
                default=0.0
            ) or 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_weighted_mid(self, levels: int = 3) -> float:
        """Calculate size-weighted mid price."""
        try:
            max_levels = min(levels, len(self.bids), len(self.asks))
            if max_levels == 0:
                return self.mid_price or 0.0
            
            bid_weighted_sum = sum(
                level.price * level.size for level in self.bids[:max_levels]
            )
            ask_weighted_sum = sum(
                level.price * level.size for level in self.asks[:max_levels]
            )
            
            bid_size_sum = sum(level.size for level in self.bids[:max_levels])
            ask_size_sum = sum(level.size for level in self.asks[:max_levels])
            
            if bid_size_sum + ask_size_sum == 0:
                return self.mid_price or 0.0
            
            return (bid_weighted_sum + ask_weighted_sum) / (bid_size_sum + ask_size_sum)
            
        except Exception:
            return self.mid_price or 0.0
    
    def _calculate_short_volatility(self) -> Optional[float]:
        """Calculate short-term realized volatility."""
        if len(self.mid_prices) < 10:  # Need minimum data points
            return None
        
        try:
            # Convert to numpy arrays
            prices = np.array(list(self.mid_prices))
            timestamps_array = np.array(list(self.timestamps))
            
            # Filter to volatility window
            current_time = timestamps_array[-1]
            window_start = current_time - (self.vol_window_secs * 1000)
            
            mask = timestamps_array >= window_start
            if np.sum(mask) < 10:
                return None
            
            windowed_prices = prices[mask]
            windowed_times = timestamps_array[mask]
            
            # Calculate log returns
            if len(windowed_prices) < 2:
                return None
            
            log_prices = np.log(windowed_prices)
            log_returns = np.diff(log_prices)
            
            if len(log_returns) == 0:
                return None
            
            # Annualized volatility
            # Assume prices are sampled roughly every 100ms
            periods_per_year = 365 * 24 * 60 * 60 * 10  # 10 samples per second
            vol = np.std(log_returns) * np.sqrt(periods_per_year)
            
            return float(vol) if np.isfinite(vol) else None
            
        except Exception as e:
            self.logger.debug("Error calculating volatility", error=str(e))
            return None
    
    def get_current_features(self) -> Optional[MicrostructureFeatures]:
        """Get current microstructure features."""
        return self.last_features
    
    def get_order_book_snapshot(self) -> Optional[OrderBookSnapshot]:
        """Get current order book snapshot."""
        if not self.bids or not self.asks:
            return None
        
        return OrderBookSnapshot(
            symbol=self.symbol,
            timestamp=self.last_update_time,
            bids=self.bids.copy(),
            asks=self.asks.copy()
        )
    
    @property
    def mid_price(self) -> Optional[float]:
        """Get current mid price."""
        if not self.bids or not self.asks:
            return None
        return (self.bids[0].price + self.asks[0].price) / 2.0
    
    @property
    def spread(self) -> Optional[float]:
        """Get current spread."""
        if not self.bids or not self.asks:
            return None
        return self.asks[0].price - self.bids[0].price
    
    @property
    def spread_bps(self) -> Optional[float]:
        """Get current spread in basis points."""
        mid = self.mid_price
        spread = self.spread
        if mid is None or spread is None or mid <= 0:
            return None
        return (spread / mid) * 10000
    
    def get_depth(self, side: OrderBookSide, levels: int = 5) -> List[OrderBookLevel]:
        """
        Get order book depth for a given side.
        
        Args:
            side: Order book side
            levels: Number of levels to return
            
        Returns:
            List of order book levels
        """
        if side == OrderBookSide.BID:
            return self.bids[:levels]
        else:
            return self.asks[:levels]
    
    def get_size_at_price(self, price: float, side: OrderBookSide) -> float:
        """
        Get total size available at or better than a given price.
        
        Args:
            price: Price level
            side: Order book side
            
        Returns:
            Total size available
        """
        total_size = 0.0
        
        if side == OrderBookSide.BID:
            for level in self.bids:
                if level.price >= price:
                    total_size += level.size
                else:
                    break
        else:
            for level in self.asks:
                if level.price <= price:
                    total_size += level.size
                else:
                    break
        
        return total_size
    
    def get_average_price_for_size(self, size: float, side: OrderBookSide) -> Optional[float]:
        """
        Get average price for executing a given size.
        
        Args:
            size: Size to execute
            side: Order book side (BUY means hitting asks, SELL means hitting bids)
            
        Returns:
            Average execution price, None if insufficient liquidity
        """
        levels = self.asks if side == OrderBookSide.ASK else self.bids
        
        remaining_size = size
        total_notional = 0.0
        
        for level in levels:
            if remaining_size <= 0:
                break
            
            level_size = min(remaining_size, level.size)
            total_notional += level_size * level.price
            remaining_size -= level_size
        
        if remaining_size > 0:
            return None  # Insufficient liquidity
        
        return total_notional / size if size > 0 else None
    
    def get_recent_features(self, n: int = 100) -> List[MicrostructureFeatures]:
        """Get recent microstructure features."""
        return list(self.features_history)[-n:]
    
    def get_volatility_percentile(self, percentile: float = 95) -> Optional[float]:
        """
        Get volatility percentile from recent history.
        
        Args:
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Volatility at given percentile
        """
        recent_features = self.get_recent_features(200)
        vols = [f.short_vol for f in recent_features if f.short_vol is not None]
        
        if len(vols) < 10:
            return None
        
        return np.percentile(vols, percentile)
    
    def is_stale(self, max_age_ms: int = 5000) -> bool:
        """
        Check if order book data is stale.
        
        Args:
            max_age_ms: Maximum age in milliseconds
            
        Returns:
            True if data is stale
        """
        if self.last_update_time == 0:
            return True
        
        current_time = timestamp_ms()
        return (current_time - self.last_update_time) > max_age_ms
    
    def reset(self) -> None:
        """Reset the cache state."""
        self.bids.clear()
        self.asks.clear()
        self.features_history.clear()
        self.mid_prices.clear()
        self.timestamps.clear()
        self.last_update_time = 0
        self.update_count = 0
        self.last_features = None
