"""
Trade data cache for market microstructure analysis.

This module provides real-time trade data management with calculations for:
- Trade flow analysis
- Volume-weighted average prices (VWAP)
- Trade size distribution
- Aggressive vs passive flow
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

from mm_bot.logging import LoggerMixin
from mm_bot.utils.safemath import safe_divide
from mm_bot.utils.time import timestamp_ms


class TradeSide(Enum):
    """Trade side from aggressor perspective."""
    BUY = "buy"   # Aggressive buy (hit ask)
    SELL = "sell"  # Aggressive sell (hit bid)
    UNKNOWN = "unknown"


@dataclass
class Trade:
    """Individual trade record."""
    symbol: str
    timestamp: int  # milliseconds
    price: float
    size: float
    side: TradeSide
    trade_id: Optional[str] = None
    
    @property
    def notional(self) -> float:
        """Get trade notional value."""
        return self.price * self.size


@dataclass
class TradeFlowMetrics:
    """Trade flow analysis metrics."""
    timestamp: int
    symbol: str
    window_ms: int
    
    # Basic metrics
    trade_count: int
    total_volume: float
    total_notional: float
    vwap: float
    
    # Flow analysis
    buy_volume: float
    sell_volume: float
    buy_notional: float
    sell_notional: float
    buy_count: int
    sell_count: int
    
    # Imbalance metrics
    volume_imbalance: float  # (buy_vol - sell_vol) / total_vol
    notional_imbalance: float  # (buy_notional - sell_notional) / total_notional
    trade_imbalance: float  # (buy_count - sell_count) / total_count
    
    # Size metrics
    avg_trade_size: float
    avg_buy_size: float
    avg_sell_size: float
    
    # Price metrics
    price_range: float
    price_std: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging/metrics."""
        return {
            "timestamp": self.timestamp,
            "trade_count": self.trade_count,
            "total_volume": self.total_volume,
            "total_notional": self.total_notional,
            "vwap": self.vwap,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
            "volume_imbalance": self.volume_imbalance,
            "notional_imbalance": self.notional_imbalance,
            "trade_imbalance": self.trade_imbalance,
            "avg_trade_size": self.avg_trade_size,
            "price_range": self.price_range,
            "price_std": self.price_std,
        }


class TradesCache(LoggerMixin):
    """
    Trade data cache with flow analysis capabilities.
    
    Maintains real-time trade data and calculates various trade flow
    metrics used for market making decisions.
    """
    
    def __init__(
        self,
        symbol: str,
        max_trades: int = 10000,
        flow_window_ms: int = 60000,  # 1 minute
        metrics_history_size: int = 1000
    ):
        """
        Initialize trades cache.
        
        Args:
            symbol: Trading symbol
            max_trades: Maximum number of trades to keep
            flow_window_ms: Flow analysis window in milliseconds
            metrics_history_size: Maximum number of metrics snapshots to keep
        """
        super().__init__()
        
        self.symbol = symbol
        self.max_trades = max_trades
        self.flow_window_ms = flow_window_ms
        
        # Trade storage
        self.trades: deque = deque(maxlen=max_trades)
        
        # Metrics history
        self.metrics_history: deque = deque(maxlen=metrics_history_size)
        self.last_metrics: Optional[TradeFlowMetrics] = None
        
        # Statistics
        self.total_trades = 0
        self.last_update_time = 0
    
    def add_trade(
        self,
        price: float,
        size: float,
        side: TradeSide,
        timestamp: Optional[int] = None,
        trade_id: Optional[str] = None
    ) -> bool:
        """
        Add a new trade to the cache.
        
        Args:
            price: Trade price
            size: Trade size
            side: Trade side (aggressor)
            timestamp: Trade timestamp in milliseconds
            trade_id: Optional trade identifier
            
        Returns:
            True if trade was added successfully
        """
        if timestamp is None:
            timestamp = timestamp_ms()
        
        try:
            trade = Trade(
                symbol=self.symbol,
                timestamp=timestamp,
                price=price,
                size=size,
                side=side,
                trade_id=trade_id
            )
            
            # Validate trade
            if not self._validate_trade(trade):
                self.logger.warning(
                    "Invalid trade received",
                    symbol=self.symbol,
                    price=price,
                    size=size,
                    side=side.value
                )
                return False
            
            self.trades.append(trade)
            self.total_trades += 1
            self.last_update_time = timestamp
            
            # Calculate updated metrics
            metrics = self._calculate_flow_metrics(timestamp)
            if metrics:
                self.metrics_history.append(metrics)
                self.last_metrics = metrics
            
            return True
            
        except Exception as e:
            self.logger.error("Error adding trade", error=str(e), symbol=self.symbol)
            return False
    
    def _validate_trade(self, trade: Trade) -> bool:
        """Validate trade data."""
        if trade.price <= 0 or trade.size <= 0:
            return False
        
        # Check for reasonable price bounds (simple sanity check)
        if len(self.trades) > 0:
            recent_prices = [t.price for t in list(self.trades)[-10:]]
            avg_price = np.mean(recent_prices)
            
            # Reject trades with price more than 50% away from recent average
            if abs(trade.price - avg_price) / avg_price > 0.5:
                return False
        
        return True
    
    def _calculate_flow_metrics(self, timestamp: int) -> Optional[TradeFlowMetrics]:
        """Calculate trade flow metrics for the current window."""
        if not self.trades:
            return None
        
        try:
            # Filter trades to window
            window_start = timestamp - self.flow_window_ms
            window_trades = [
                trade for trade in self.trades
                if trade.timestamp >= window_start
            ]
            
            if not window_trades:
                return None
            
            # Basic metrics
            trade_count = len(window_trades)
            prices = [t.price for t in window_trades]
            sizes = [t.size for t in window_trades]
            notionals = [t.notional for t in window_trades]
            
            total_volume = sum(sizes)
            total_notional = sum(notionals)
            vwap = total_notional / total_volume if total_volume > 0 else 0
            
            # Flow analysis
            buy_trades = [t for t in window_trades if t.side == TradeSide.BUY]
            sell_trades = [t for t in window_trades if t.side == TradeSide.SELL]
            
            buy_volume = sum(t.size for t in buy_trades)
            sell_volume = sum(t.size for t in sell_trades)
            buy_notional = sum(t.notional for t in buy_trades)
            sell_notional = sum(t.notional for t in sell_trades)
            buy_count = len(buy_trades)
            sell_count = len(sell_trades)
            
            # Imbalance metrics
            volume_imbalance = safe_divide(
                buy_volume - sell_volume,
                total_volume,
                default=0.0
            ) or 0.0
            
            notional_imbalance = safe_divide(
                buy_notional - sell_notional,
                total_notional,
                default=0.0
            ) or 0.0
            
            trade_imbalance = safe_divide(
                buy_count - sell_count,
                trade_count,
                default=0.0
            ) or 0.0
            
            # Size metrics
            avg_trade_size = total_volume / trade_count if trade_count > 0 else 0
            avg_buy_size = buy_volume / buy_count if buy_count > 0 else 0
            avg_sell_size = sell_volume / sell_count if sell_count > 0 else 0
            
            # Price metrics
            price_range = max(prices) - min(prices) if prices else 0
            price_std = np.std(prices) if len(prices) > 1 else 0
            
            return TradeFlowMetrics(
                timestamp=timestamp,
                symbol=self.symbol,
                window_ms=self.flow_window_ms,
                trade_count=trade_count,
                total_volume=total_volume,
                total_notional=total_notional,
                vwap=vwap,
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                buy_notional=buy_notional,
                sell_notional=sell_notional,
                buy_count=buy_count,
                sell_count=sell_count,
                volume_imbalance=volume_imbalance,
                notional_imbalance=notional_imbalance,
                trade_imbalance=trade_imbalance,
                avg_trade_size=avg_trade_size,
                avg_buy_size=avg_buy_size,
                avg_sell_size=avg_sell_size,
                price_range=price_range,
                price_std=float(price_std),
            )
            
        except Exception as e:
            self.logger.error("Error calculating flow metrics", error=str(e))
            return None
    
    def get_current_metrics(self) -> Optional[TradeFlowMetrics]:
        """Get current trade flow metrics."""
        return self.last_metrics
    
    def get_recent_trades(self, n: int = 100) -> List[Trade]:
        """Get recent trades."""
        return list(self.trades)[-n:]
    
    def get_trades_in_window(self, window_ms: int) -> List[Trade]:
        """
        Get trades within a time window.
        
        Args:
            window_ms: Window size in milliseconds
            
        Returns:
            List of trades within the window
        """
        if not self.trades:
            return []
        
        current_time = self.last_update_time or timestamp_ms()
        window_start = current_time - window_ms
        
        return [
            trade for trade in self.trades
            if trade.timestamp >= window_start
        ]
    
    def get_vwap(self, window_ms: Optional[int] = None) -> Optional[float]:
        """
        Calculate volume-weighted average price.
        
        Args:
            window_ms: Time window in milliseconds (None for all trades)
            
        Returns:
            VWAP or None if no trades
        """
        if window_ms is None:
            trades = list(self.trades)
        else:
            trades = self.get_trades_in_window(window_ms)
        
        if not trades:
            return None
        
        total_notional = sum(t.notional for t in trades)
        total_volume = sum(t.size for t in trades)
        
        return total_notional / total_volume if total_volume > 0 else None
    
    def get_trade_rate(self, window_ms: int = 60000) -> float:
        """
        Get trade rate (trades per second).
        
        Args:
            window_ms: Window size in milliseconds
            
        Returns:
            Trades per second
        """
        trades = self.get_trades_in_window(window_ms)
        if not trades:
            return 0.0
        
        return len(trades) / (window_ms / 1000.0)
    
    def get_volume_profile(
        self,
        price_buckets: int = 20,
        window_ms: Optional[int] = None
    ) -> Dict[float, float]:
        """
        Get volume profile (volume by price level).
        
        Args:
            price_buckets: Number of price buckets
            window_ms: Time window (None for all trades)
            
        Returns:
            Dictionary mapping price levels to volumes
        """
        if window_ms is None:
            trades = list(self.trades)
        else:
            trades = self.get_trades_in_window(window_ms)
        
        if not trades:
            return {}
        
        prices = [t.price for t in trades]
        volumes = [t.size for t in trades]
        
        min_price = min(prices)
        max_price = max(prices)
        
        if min_price == max_price:
            return {min_price: sum(volumes)}
        
        bucket_size = (max_price - min_price) / price_buckets
        profile = {}
        
        for trade in trades:
            bucket_idx = min(
                int((trade.price - min_price) / bucket_size),
                price_buckets - 1
            )
            bucket_price = min_price + (bucket_idx + 0.5) * bucket_size
            profile[bucket_price] = profile.get(bucket_price, 0) + trade.size
        
        return profile
    
    def get_size_distribution(
        self,
        percentiles: List[float] = [25, 50, 75, 90, 95, 99],
        window_ms: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get trade size distribution.
        
        Args:
            percentiles: Percentiles to calculate
            window_ms: Time window (None for all trades)
            
        Returns:
            Dictionary with size distribution statistics
        """
        if window_ms is None:
            trades = list(self.trades)
        else:
            trades = self.get_trades_in_window(window_ms)
        
        if not trades:
            return {}
        
        sizes = [t.size for t in trades]
        
        result = {
            "mean": np.mean(sizes),
            "std": np.std(sizes),
            "min": np.min(sizes),
            "max": np.max(sizes),
        }
        
        for p in percentiles:
            result[f"p{p}"] = np.percentile(sizes, p)
        
        return result
    
    def get_flow_toxicity(self, window_ms: int = 300000) -> Optional[float]:
        """
        Calculate flow toxicity (adverse selection measure).
        
        This is a simplified measure based on price impact after trades.
        
        Args:
            window_ms: Window size in milliseconds
            
        Returns:
            Flow toxicity measure (0-1, higher = more toxic)
        """
        trades = self.get_trades_in_window(window_ms)
        if len(trades) < 10:
            return None
        
        try:
            # Calculate price impact for each trade
            impacts = []
            
            for i, trade in enumerate(trades[:-5]):  # Exclude last few trades
                # Look at price movement in next 5 trades
                future_trades = trades[i+1:i+6]
                if not future_trades:
                    continue
                
                future_prices = [t.price for t in future_trades]
                future_vwap = np.mean(future_prices)
                
                # Calculate impact based on trade direction
                if trade.side == TradeSide.BUY:
                    impact = (future_vwap - trade.price) / trade.price
                else:
                    impact = (trade.price - future_vwap) / trade.price
                
                impacts.append(max(0, impact))  # Only consider adverse moves
            
            if not impacts:
                return None
            
            return min(1.0, np.mean(impacts) * 100)  # Scale to 0-1
            
        except Exception:
            return None
    
    def is_stale(self, max_age_ms: int = 30000) -> bool:
        """
        Check if trade data is stale.
        
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
        self.trades.clear()
        self.metrics_history.clear()
        self.last_metrics = None
        self.total_trades = 0
        self.last_update_time = 0
