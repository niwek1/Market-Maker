"""
Prometheus metrics collection and monitoring.
"""

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    start_http_server,
    CollectorRegistry,
    REGISTRY,
)

from mm_bot.config import MetricsConfig


class MetricsCollector:
    """Centralized metrics collection for the market making bot."""
    
    def __init__(self, config: MetricsConfig, registry: Optional[CollectorRegistry] = None):
        self.config = config
        self.registry = registry or REGISTRY
        self.enabled = config.enabled
        
        if not self.enabled:
            return
        
        # Trading metrics
        self.trades_total = Counter(
            "mm_trades_total",
            "Total number of trades executed",
            ["symbol", "side", "exchange"],
            registry=self.registry
        )
        
        self.trade_volume = Counter(
            "mm_trade_volume_total",
            "Total trading volume",
            ["symbol", "exchange"],
            registry=self.registry
        )
        
        self.trade_fees = Counter(
            "mm_trade_fees_total",
            "Total trading fees paid",
            ["symbol", "exchange"],
            registry=self.registry
        )
        
        # PnL metrics
        self.unrealized_pnl = Gauge(
            "mm_unrealized_pnl",
            "Current unrealized PnL",
            ["symbol", "exchange"],
            registry=self.registry
        )
        
        self.realized_pnl = Gauge(
            "mm_realized_pnl",
            "Current realized PnL",
            ["symbol", "exchange"],
            registry=self.registry
        )
        
        self.total_pnl = Gauge(
            "mm_total_pnl",
            "Total PnL (realized + unrealized)",
            ["symbol", "exchange"],
            registry=self.registry
        )
        
        # Position metrics
        self.position_size = Gauge(
            "mm_position_size",
            "Current position size",
            ["symbol", "exchange"],
            registry=self.registry
        )
        
        self.position_notional = Gauge(
            "mm_position_notional",
            "Current position notional value",
            ["symbol", "exchange"],
            registry=self.registry
        )
        
        # Order metrics
        self.orders_placed = Counter(
            "mm_orders_placed_total",
            "Total orders placed",
            ["symbol", "side", "type", "exchange"],
            registry=self.registry
        )
        
        self.orders_filled = Counter(
            "mm_orders_filled_total",
            "Total orders filled",
            ["symbol", "side", "type", "exchange"],
            registry=self.registry
        )
        
        self.orders_cancelled = Counter(
            "mm_orders_cancelled_total",
            "Total orders cancelled",
            ["symbol", "side", "exchange"],
            registry=self.registry
        )
        
        self.active_orders = Gauge(
            "mm_active_orders",
            "Number of active orders",
            ["symbol", "side", "exchange"],
            registry=self.registry
        )
        
        # Market data metrics
        self.bid_price = Gauge(
            "mm_bid_price",
            "Current bid price",
            ["symbol", "exchange"],
            registry=self.registry
        )
        
        self.ask_price = Gauge(
            "mm_ask_price",
            "Current ask price",
            ["symbol", "exchange"],
            registry=self.registry
        )
        
        self.spread_bps = Gauge(
            "mm_spread_bps",
            "Current spread in basis points",
            ["symbol", "exchange"],
            registry=self.registry
        )
        
        self.order_book_imbalance = Gauge(
            "mm_order_book_imbalance",
            "Order book imbalance (-1 to 1)",
            ["symbol", "exchange"],
            registry=self.registry
        )
        
        self.volatility = Gauge(
            "mm_volatility",
            "Current volatility estimate",
            ["symbol", "exchange"],
            registry=self.registry
        )
        
        # Strategy metrics
        self.reservation_price = Gauge(
            "mm_reservation_price",
            "Avellaneda-Stoikov reservation price",
            ["symbol", "exchange"],
            registry=self.registry
        )
        
        self.optimal_spread = Gauge(
            "mm_optimal_spread",
            "Avellaneda-Stoikov optimal spread",
            ["symbol", "exchange"],
            registry=self.registry
        )
        
        self.inventory_skew = Gauge(
            "mm_inventory_skew",
            "Current inventory skew",
            ["symbol", "exchange"],
            registry=self.registry
        )
        
        # Risk metrics
        self.risk_limit_utilization = Gauge(
            "mm_risk_limit_utilization",
            "Risk limit utilization (0-1)",
            ["limit_type", "symbol", "exchange"],
            registry=self.registry
        )
        
        self.drawdown_pct = Gauge(
            "mm_drawdown_pct",
            "Current drawdown percentage",
            ["exchange"],
            registry=self.registry
        )
        
        self.var_estimate = Gauge(
            "mm_var_estimate",
            "Value at Risk estimate",
            ["symbol", "confidence", "exchange"],
            registry=self.registry
        )
        
        # Performance metrics
        self.fill_ratio = Gauge(
            "mm_fill_ratio",
            "Order fill ratio",
            ["symbol", "side", "exchange"],
            registry=self.registry
        )
        
        self.adverse_selection = Gauge(
            "mm_adverse_selection",
            "Adverse selection measure",
            ["symbol", "exchange"],
            registry=self.registry
        )
        
        self.realized_spread_bps = Gauge(
            "mm_realized_spread_bps",
            "Realized spread in basis points",
            ["symbol", "exchange"],
            registry=self.registry
        )
        
        # Latency metrics
        self.order_latency = Histogram(
            "mm_order_latency_seconds",
            "Order placement latency",
            ["exchange", "action"],
            registry=self.registry
        )
        
        self.market_data_latency = Histogram(
            "mm_market_data_latency_seconds",
            "Market data processing latency",
            ["exchange", "data_type"],
            registry=self.registry
        )
        
        # System metrics
        self.uptime_seconds = Gauge(
            "mm_uptime_seconds",
            "Bot uptime in seconds",
            registry=self.registry
        )
        
        self.quote_updates = Counter(
            "mm_quote_updates_total",
            "Total quote updates",
            ["symbol", "exchange"],
            registry=self.registry
        )
        
        self.errors_total = Counter(
            "mm_errors_total",
            "Total errors encountered",
            ["error_type", "component"],
            registry=self.registry
        )
        
        # System info
        self.info = Info(
            "mm_bot_info",
            "Market making bot information",
            registry=self.registry
        )
        
        # Initialize tracking variables
        self.start_time = time.time()
        self.last_update = time.time()
        
        # Performance tracking
        self.performance_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        fee: float,
        exchange: str = "default"
    ) -> None:
        """Record a trade execution."""
        if not self.enabled:
            return
        
        self.trades_total.labels(symbol=symbol, side=side, exchange=exchange).inc()
        self.trade_volume.labels(symbol=symbol, exchange=exchange).inc(size * price)
        self.trade_fees.labels(symbol=symbol, exchange=exchange).inc(fee)
    
    def update_pnl(
        self,
        symbol: str,
        unrealized: float,
        realized: float,
        exchange: str = "default"
    ) -> None:
        """Update PnL metrics."""
        if not self.enabled:
            return
        
        self.unrealized_pnl.labels(symbol=symbol, exchange=exchange).set(unrealized)
        self.realized_pnl.labels(symbol=symbol, exchange=exchange).set(realized)
        self.total_pnl.labels(symbol=symbol, exchange=exchange).set(unrealized + realized)
    
    def update_position(
        self,
        symbol: str,
        size: float,
        notional: float,
        exchange: str = "default"
    ) -> None:
        """Update position metrics."""
        if not self.enabled:
            return
        
        self.position_size.labels(symbol=symbol, exchange=exchange).set(size)
        self.position_notional.labels(symbol=symbol, exchange=exchange).set(notional)
    
    def record_order(
        self,
        action: str,  # placed, filled, cancelled
        symbol: str,
        side: str,
        order_type: str = "limit",
        exchange: str = "default"
    ) -> None:
        """Record order action."""
        if not self.enabled:
            return
        
        if action == "placed":
            self.orders_placed.labels(
                symbol=symbol, side=side, type=order_type, exchange=exchange
            ).inc()
        elif action == "filled":
            self.orders_filled.labels(
                symbol=symbol, side=side, type=order_type, exchange=exchange
            ).inc()
        elif action == "cancelled":
            self.orders_cancelled.labels(
                symbol=symbol, side=side, exchange=exchange
            ).inc()
    
    def update_active_orders(
        self,
        symbol: str,
        bid_count: int,
        ask_count: int,
        exchange: str = "default"
    ) -> None:
        """Update active order counts."""
        if not self.enabled:
            return
        
        self.active_orders.labels(symbol=symbol, side="bid", exchange=exchange).set(bid_count)
        self.active_orders.labels(symbol=symbol, side="ask", exchange=exchange).set(ask_count)
    
    def update_market_data(
        self,
        symbol: str,
        bid: float,
        ask: float,
        spread_bps: float,
        imbalance: float,
        volatility: float,
        exchange: str = "default"
    ) -> None:
        """Update market data metrics."""
        if not self.enabled:
            return
        
        self.bid_price.labels(symbol=symbol, exchange=exchange).set(bid)
        self.ask_price.labels(symbol=symbol, exchange=exchange).set(ask)
        self.spread_bps.labels(symbol=symbol, exchange=exchange).set(spread_bps)
        self.order_book_imbalance.labels(symbol=symbol, exchange=exchange).set(imbalance)
        self.volatility.labels(symbol=symbol, exchange=exchange).set(volatility)
    
    def update_strategy_metrics(
        self,
        symbol: str,
        reservation_price: float,
        optimal_spread: float,
        inventory_skew: float,
        exchange: str = "default"
    ) -> None:
        """Update strategy-specific metrics."""
        if not self.enabled:
            return
        
        self.reservation_price.labels(symbol=symbol, exchange=exchange).set(reservation_price)
        self.optimal_spread.labels(symbol=symbol, exchange=exchange).set(optimal_spread)
        self.inventory_skew.labels(symbol=symbol, exchange=exchange).set(inventory_skew)
    
    def update_risk_metrics(
        self,
        limit_utilization: Dict[str, float],
        drawdown_pct: float,
        symbol: str = "",
        exchange: str = "default"
    ) -> None:
        """Update risk metrics."""
        if not self.enabled:
            return
        
        for limit_type, utilization in limit_utilization.items():
            self.risk_limit_utilization.labels(
                limit_type=limit_type, symbol=symbol, exchange=exchange
            ).set(utilization)
        
        self.drawdown_pct.labels(exchange=exchange).set(drawdown_pct)
    
    def record_latency(
        self,
        operation: str,
        latency_seconds: float,
        exchange: str = "default",
        data_type: str = ""
    ) -> None:
        """Record operation latency."""
        if not self.enabled:
            return
        
        if operation in ["place_order", "cancel_order", "modify_order"]:
            self.order_latency.labels(exchange=exchange, action=operation).observe(latency_seconds)
        elif data_type:
            self.market_data_latency.labels(
                exchange=exchange, data_type=data_type
            ).observe(latency_seconds)
    
    def record_error(self, error_type: str, component: str) -> None:
        """Record an error occurrence."""
        if not self.enabled:
            return
        
        self.errors_total.labels(error_type=error_type, component=component).inc()
    
    def update_performance_metrics(
        self,
        symbol: str,
        fill_ratio: float,
        adverse_selection: float,
        realized_spread_bps: float,
        exchange: str = "default"
    ) -> None:
        """Update performance metrics."""
        if not self.enabled:
            return
        
        self.fill_ratio.labels(symbol=symbol, side="all", exchange=exchange).set(fill_ratio)
        self.adverse_selection.labels(symbol=symbol, exchange=exchange).set(adverse_selection)
        self.realized_spread_bps.labels(symbol=symbol, exchange=exchange).set(realized_spread_bps)
    
    def update_system_metrics(self) -> None:
        """Update system-level metrics."""
        if not self.enabled:
            return
        
        current_time = time.time()
        self.uptime_seconds.set(current_time - self.start_time)
        self.last_update = current_time
    
    def set_bot_info(self, info: Dict[str, str]) -> None:
        """Set bot information."""
        if not self.enabled:
            return
        
        self.info.info(info)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for monitoring."""
        if not self.enabled:
            return {}
        
        return {
            "uptime_seconds": time.time() - self.start_time,
            "last_update": self.last_update,
            "metrics_enabled": self.enabled,
        }


def setup_metrics(config: MetricsConfig) -> Optional[MetricsCollector]:
    """
    Set up metrics collection and HTTP server.
    
    Args:
        config: Metrics configuration
        
    Returns:
        MetricsCollector instance if enabled, None otherwise
    """
    if not config.enabled:
        return None
    
    # Create metrics collector
    collector = MetricsCollector(config)
    
    # Start HTTP server for Prometheus scraping
    start_http_server(config.port)
    
    # Set initial bot info
    collector.set_bot_info({
        "version": "0.1.0",
        "mode": "unknown",  # Will be set by main application
        "start_time": str(int(time.time())),
    })
    
    return collector


# Global metrics instance (will be set by setup_metrics)
metrics: Optional[MetricsCollector] = None


def get_metrics() -> Optional[MetricsCollector]:
    """Get the global metrics collector instance."""
    return metrics


def set_metrics(collector: Optional[MetricsCollector]) -> None:
    """Set the global metrics collector instance."""
    global metrics
    metrics = collector
