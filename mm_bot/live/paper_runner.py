"""
Complete paper trading runner that integrates all components.

This module provides a comprehensive paper trading system that combines:
- Exchange connectivity and market data
- Avellaneda-Stoikov strategy
- Risk management and position tracking
- Order management with realistic simulation
- Performance monitoring and reporting
"""

import asyncio
import signal
import time
from typing import Dict, List, Optional, Any

from mm_bot.config import Config
from mm_bot.logging import LoggerMixin, get_logger
from mm_bot.metrics import get_metrics
from mm_bot.marketdata.l2_cache import L2Cache
from mm_bot.marketdata.trades_cache import TradesCache
from mm_bot.strategy.avellaneda_stoikov import AvellanedaStoikovStrategy
from mm_bot.strategy.signals import SignalAggregator
from mm_bot.strategy.quoting import QuoteAdjuster, QuoteConstraints
from mm_bot.execution.order_manager import OrderManager, OrderSide, OrderType
from mm_bot.risk.limits import RiskLimitManager
from mm_bot.risk.kill_switch import get_global_kill_switch
from mm_bot.exchanges.binance import BinanceAdapter
from mm_bot.exchanges.kraken import KrakenAdapter
from mm_bot.exchanges.coinbase import CoinbaseAdapter
from mm_bot.exchanges.hyperliquid_official import HyperliquidOfficialAdapter


class PaperTradingBot(LoggerMixin):
    """
    Complete paper trading bot implementation.
    
    Features:
    - Multi-exchange support with unified interface
    - Real-time market data processing
    - Avellaneda-Stoikov strategy with microstructure signals
    - Comprehensive risk management
    - Performance tracking and reporting
    - Graceful shutdown and error recovery
    """
    
    def __init__(self, config: Config):
        """
        Initialize paper trading bot.
        
        Args:
            config: Bot configuration
        """
        super().__init__()
        self.config = config
        self.running = False
        
        # Core components
        self.exchange_adapter = None
        self.order_manager = None
        self.risk_manager = None
        self.strategy = None
        self.signal_aggregator = None
        self.quote_adjuster = None
        
        # Market data
        self.l2_caches: Dict[str, L2Cache] = {}
        self.trade_caches: Dict[str, TradesCache] = {}
        
        # Performance tracking
        self.start_time = 0
        self.total_orders = 0
        self.total_fills = 0
        self.total_volume = 0.0
        self.total_pnl = 0.0
        
        # Position tracking
        self.positions: Dict[str, float] = {}
        self.balances: Dict[str, float] = {}
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Kill switch integration
        self.kill_switch = get_global_kill_switch()
        
        self.logger.info(
            "Initialized paper trading bot",
            mode=config.mode,
            exchange=config.exchange,
            symbols=config.symbols
        )
    
    async def start(self) -> None:
        """Start the paper trading bot."""
        try:
            if self.running:
                self.logger.warning("Bot already running")
                return
            
            self.logger.info("Starting paper trading bot")
            self.start_time = time.time()
            self.running = True
            
            # Initialize components
            await self._initialize_components()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info("Paper trading bot started successfully")
            
            # Main event loop
            await self._main_loop()
            
        except Exception as e:
            self.logger.error("Error starting paper trading bot", error=str(e))
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the paper trading bot."""
        try:
            if not self.running:
                return
            
            self.logger.info("Stopping paper trading bot")
            self.running = False
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Stop components
            if self.order_manager:
                await self.order_manager.stop()
            
            if self.exchange_adapter:
                await self.exchange_adapter.disconnect()
            
            # Final performance report
            await self._generate_final_report()
            
            self.logger.info("Paper trading bot stopped")
            
        except Exception as e:
            self.logger.error("Error stopping paper trading bot", error=str(e))
    
    async def _initialize_components(self) -> None:
        """Initialize all bot components."""
        # Initialize exchange adapter
        self.exchange_adapter = await self._create_exchange_adapter()
        if not await self.exchange_adapter.connect():
            raise RuntimeError("Failed to connect to exchange")
        
        # Initialize order manager
        self.order_manager = OrderManager(self.config.execution, self.exchange_adapter)
        await self.order_manager.start()
        
        # Initialize risk manager
        self.risk_manager = RiskLimitManager(self.config.risk)
        
        # Initialize strategy
        self.strategy = AvellanedaStoikovStrategy(self.config.strategy.as_)
        
        # Initialize signal aggregator
        self.signal_aggregator = SignalAggregator()
        
        # Initialize quote adjuster
        self.quote_adjuster = QuoteAdjuster()
        
        # Initialize market data caches
        for symbol in self.config.symbols:
            self.l2_caches[symbol] = L2Cache(
                symbol,
                vol_window_secs=self.config.strategy.as_.vol_window_secs,
                imbalance_window_ms=self.config.strategy.as_.imbalance_window_ms
            )
            self.trade_caches[symbol] = TradesCache(symbol)
            self.positions[symbol] = 0.0
        
        # Setup callbacks
        self._setup_callbacks()
        
        # Initialize balances (simulated)
        self.balances = {
            'BTC': 1.0,
            'USDT': 50000.0,
            'ETH': 10.0,
        }
        
        self.logger.info("All components initialized")
    
    async def _create_exchange_adapter(self):
        """Create appropriate exchange adapter."""
        exchange_config = self.config.get_exchange_config(self.config.exchange)
        fee_config = self.config.get_fee_config(self.config.exchange)
        
        if self.config.exchange.lower() == 'binance':
            return BinanceAdapter(exchange_config, fee_config)
        elif self.config.exchange.lower() == 'kraken':
            return KrakenAdapter(exchange_config, fee_config)
        elif self.config.exchange.lower() == 'coinbase':
            return CoinbaseAdapter(exchange_config, fee_config)
        elif self.config.exchange.lower() == 'hyperliquid':
            # Use official SDK for REAL trading capabilities
            return HyperliquidOfficialAdapter(exchange_config, fee_config)
        else:
            raise ValueError(f"Unsupported exchange: {self.config.exchange}")
    
    def _setup_callbacks(self) -> None:
        """Setup callbacks for various components."""
        # Order manager callbacks
        self.order_manager.add_fill_callback(self._handle_order_fill)
        self.order_manager.add_cancel_callback(self._handle_order_cancel)
        self.order_manager.add_reject_callback(self._handle_order_reject)
        
        # Risk manager callbacks
        self.risk_manager.add_warning_callback(self._handle_risk_warning)
        self.risk_manager.add_breach_callback(self._handle_risk_breach)
        
        # Kill switch callbacks
        self.kill_switch.add_shutdown_callback(self._handle_emergency_shutdown)
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info("Received shutdown signal", signal=signum)
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks."""
        tasks = [
            self._market_data_loop(),
            self._strategy_loop(),
            self._risk_monitoring_loop(),
            self._performance_reporting_loop(),
        ]
        
        for task_func in tasks:
            task = asyncio.create_task(task_func)
            self.background_tasks.append(task)
        
        self.logger.info("Started background tasks", count=len(tasks))
    
    async def _main_loop(self) -> None:
        """Main event loop."""
        while self.running and not self.kill_switch.is_triggered():
            try:
                # Check if we should continue running
                if not self.running:
                    break
                
                # Health checks
                await self._health_check()
                
                # Sleep briefly
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error("Error in main loop", error=str(e))
                await asyncio.sleep(1)
        
        self.logger.info("Exiting main loop")
    
    async def _market_data_loop(self) -> None:
        """Market data processing loop."""
        while self.running:
            try:
                for symbol in self.config.symbols:
                    # Fetch order book
                    order_book = await self.exchange_adapter.fetch_order_book(symbol, 20)
                    
                    # Update L2 cache
                    l2_cache = self.l2_caches[symbol]
                    bids = [(level.price, level.size) for level in order_book.bids]
                    asks = [(level.price, level.size) for level in order_book.asks]
                    l2_cache.update_order_book(bids, asks, order_book.timestamp)
                    
                    # Update metrics
                    metrics = get_metrics()
                    if metrics and l2_cache.last_features:
                        features = l2_cache.last_features
                        metrics.update_market_data(
                            symbol=symbol,
                            bid=features.bid,
                            ask=features.ask,
                            spread_bps=features.spread_bps,
                            imbalance=features.volume_imbalance,
                            volatility=features.short_vol or 0.0
                        )
                
                await asyncio.sleep(0.1)  # 100ms market data updates
                
            except Exception as e:
                self.logger.error("Error in market data loop", error=str(e))
                await asyncio.sleep(1)
    
    async def _strategy_loop(self) -> None:
        """Strategy execution loop."""
        while self.running:
            try:
                for symbol in self.config.symbols:
                    await self._execute_strategy_for_symbol(symbol)
                
                await asyncio.sleep(1)  # 1 second strategy updates
                
            except Exception as e:
                self.logger.error("Error in strategy loop", error=str(e))
                await asyncio.sleep(5)
    
    async def _execute_strategy_for_symbol(self, symbol: str) -> None:
        """Execute strategy for a specific symbol."""
        try:
            l2_cache = self.l2_caches[symbol]
            features = l2_cache.get_current_features()
            
            if not features or l2_cache.is_stale():
                return
            
            # Generate signals
            trade_metrics = self.trade_caches[symbol].get_current_metrics()
            signals = self.signal_aggregator.calculate_composite_signal(features, trade_metrics)
            
            # Get current position and calculate inventory
            current_position = self.positions[symbol]
            max_position = self.config.risk.max_position
            
            # Generate AS quote
            quote = self.strategy.generate_quote(
                mid_price=features.mid,
                volatility=features.short_vol or 0.02,
                inventory=current_position,
                max_inventory=max_position,
                quote_size=self.config.risk.quote_size_base
            )
            
            # Apply signal adjustments
            if signals.combined_confidence > 0.5:
                # Adjust prices based on signals
                signal_adjustment = signals.combined_value * features.mid * 0.001  # 0.1% max adjustment
                quote.bid_price -= signal_adjustment
                quote.ask_price += signal_adjustment
            
            # Adjust quote for exchange constraints
            constraints = QuoteConstraints(
                min_spread_bps=self.config.risk.min_spread_bps,
                max_spread_bps=self.config.risk.max_spread_bps,
                tick_size=self.exchange_adapter.get_tick_size(symbol),
                lot_size=self.exchange_adapter.get_lot_size(symbol),
                min_notional=self.exchange_adapter.get_min_notional(symbol)
            )
            
            adjusted_quote = self.quote_adjuster.adjust_quote(
                quote, constraints, features.mid, self.balances.get('USDT', 0)
            )
            
            if not adjusted_quote.valid:
                self.logger.debug("Quote rejected", symbol=symbol, reason=adjusted_quote.rejection_reason)
                return
            
            # Check risk limits
            allowed, violations = self.risk_manager.check_order_allowed(
                symbol, "buy", adjusted_quote.bid_size, adjusted_quote.bid_price
            )
            
            if not allowed:
                self.logger.warning("Orders blocked by risk limits", symbol=symbol, violations=violations)
                return
            
            # Cancel existing orders
            await self.order_manager.cancel_orders_by_symbol(symbol, "strategy_update")
            
            # Place new orders
            await self._place_quotes(symbol, adjusted_quote)
            
        except Exception as e:
            self.logger.error("Error executing strategy", symbol=symbol, error=str(e))
    
    async def _place_quotes(self, symbol: str, quote) -> None:
        """Place bid and ask quotes."""
        try:
            # Place bid order
            if quote.bid_size > 0:
                await self.order_manager.place_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    amount=quote.bid_size,
                    price=quote.bid_price,
                    order_type=OrderType.LIMIT,
                    strategy_id="avellaneda_stoikov"
                )
            
            # Place ask order
            if quote.ask_size > 0:
                await self.order_manager.place_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    amount=quote.ask_size,
                    price=quote.ask_price,
                    order_type=OrderType.LIMIT,
                    strategy_id="avellaneda_stoikov"
                )
            
            self.logger.debug(
                "Placed quotes",
                symbol=symbol,
                bid=f"{quote.bid_price:.2f}@{quote.bid_size:.4f}",
                ask=f"{quote.ask_price:.2f}@{quote.ask_size:.4f}",
                spread_bps=quote.spread_bps
            )
            
        except Exception as e:
            self.logger.error("Error placing quotes", symbol=symbol, error=str(e))
    
    async def _risk_monitoring_loop(self) -> None:
        """Risk monitoring loop."""
        while self.running:
            try:
                # Update risk metrics
                for symbol in self.config.symbols:
                    position = self.positions[symbol]
                    l2_cache = self.l2_caches[symbol]
                    features = l2_cache.get_current_features()
                    
                    if features:
                        # Update position limits
                        self.risk_manager.update_position(symbol, position, features.mid)
                        
                        # Update PnL (simplified calculation)
                        unrealized_pnl = position * features.mid  # This would be more complex in reality
                        self.risk_manager.update_pnl(unrealized_pnl)
                
                # Check kill switch triggers
                risk_summary = self.risk_manager.get_risk_summary()
                if risk_summary['current_drawdown_pct'] > 0:
                    self.kill_switch.check_drawdown_trigger(risk_summary['current_drawdown_pct'])
                
                await asyncio.sleep(5)  # 5 second risk checks
                
            except Exception as e:
                self.logger.error("Error in risk monitoring", error=str(e))
                await asyncio.sleep(10)
    
    async def _performance_reporting_loop(self) -> None:
        """Performance reporting loop."""
        while self.running:
            try:
                # Generate performance report
                report = await self._generate_performance_report()
                
                self.logger.info(
                    "Performance update",
                    uptime_minutes=(time.time() - self.start_time) / 60,
                    total_orders=self.total_orders,
                    total_fills=self.total_fills,
                    total_volume=self.total_volume,
                    total_pnl=self.total_pnl
                )
                
                await asyncio.sleep(60)  # 1 minute performance reports
                
            except Exception as e:
                self.logger.error("Error in performance reporting", error=str(e))
                await asyncio.sleep(60)
    
    async def _health_check(self) -> None:
        """Perform health checks."""
        # Check exchange connection
        if self.exchange_adapter.status.value != "connected":
            self.logger.warning("Exchange not connected", status=self.exchange_adapter.status.value)
        
        # Check for stale market data
        for symbol, l2_cache in self.l2_caches.items():
            if l2_cache.is_stale(max_age_ms=10000):  # 10 seconds
                self.logger.warning("Stale market data", symbol=symbol)
        
        # Record connection for kill switch
        self.kill_switch.record_connection()
    
    def _handle_order_fill(self, order) -> None:
        """Handle order fill."""
        self.total_fills += 1
        self.total_volume += order.filled_amount * (order.average_price or order.price or 0)
        
        # Update position
        if order.symbol in self.positions:
            if order.is_buy:
                self.positions[order.symbol] += order.filled_amount
            else:
                self.positions[order.symbol] -= order.filled_amount
        
        # Update metrics
        metrics = get_metrics()
        if metrics:
            metrics.record_trade(
                symbol=order.symbol,
                side=order.side.value,
                size=order.filled_amount,
                price=order.average_price or order.price or 0,
                fee=order.fee
            )
        
        self.logger.info(
            "Order filled",
            symbol=order.symbol,
            side=order.side.value,
            filled=order.filled_amount,
            price=order.average_price,
            new_position=self.positions.get(order.symbol, 0)
        )
    
    def _handle_order_cancel(self, order) -> None:
        """Handle order cancellation."""
        self.logger.debug("Order cancelled", order_id=order.id, symbol=order.symbol)
    
    def _handle_order_reject(self, order, reason: str) -> None:
        """Handle order rejection."""
        self.logger.warning("Order rejected", order_id=order.id, reason=reason)
    
    def _handle_risk_warning(self, limit) -> None:
        """Handle risk limit warning."""
        self.logger.warning(
            "Risk limit warning",
            limit_type=limit.limit_type.value,
            utilization=limit.utilization,
            symbol=limit.symbol
        )
    
    def _handle_risk_breach(self, limit) -> None:
        """Handle risk limit breach."""
        self.logger.error(
            "Risk limit breach",
            limit_type=limit.limit_type.value,
            current_value=limit.current_value,
            max_value=limit.max_value,
            symbol=limit.symbol
        )
        
        # Cancel all orders on breach
        asyncio.create_task(self.order_manager.cancel_all_orders("risk_breach"))
    
    async def _handle_emergency_shutdown(self) -> None:
        """Handle emergency shutdown."""
        self.logger.critical("Emergency shutdown initiated")
        
        # Cancel all orders
        if self.order_manager:
            await self.order_manager.cancel_all_orders("emergency_shutdown")
        
        # Stop the bot
        self.running = False
    
    async def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        uptime = time.time() - self.start_time
        
        report = {
            "uptime_seconds": uptime,
            "total_orders": self.total_orders,
            "total_fills": self.total_fills,
            "fill_ratio": self.total_fills / max(self.total_orders, 1),
            "total_volume": self.total_volume,
            "total_pnl": self.total_pnl,
            "positions": dict(self.positions),
            "balances": dict(self.balances),
            "risk_summary": self.risk_manager.get_risk_summary() if self.risk_manager else {},
            "order_manager_stats": self.order_manager.get_statistics() if self.order_manager else {},
        }
        
        return report
    
    async def _generate_final_report(self) -> None:
        """Generate final performance report."""
        try:
            report = await self._generate_performance_report()
            
            self.logger.info(
                "Final performance report",
                **{k: v for k, v in report.items() if not isinstance(v, dict)}
            )
            
            # Save detailed report
            # In a real implementation, this would save to a file or database
            
        except Exception as e:
            self.logger.error("Error generating final report", error=str(e))
