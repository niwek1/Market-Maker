"""
Live trading runner (placeholder for actual live trading implementation).

This module would contain the live trading implementation, which is similar
to the paper trading runner but with real money and additional safety checks.
"""

from mm_bot.live.paper_runner import PaperTradingBot
from mm_bot.logging import LoggerMixin


class LiveTradingBot(PaperTradingBot):
    """
    Live trading bot implementation.
    
    Extends PaperTradingBot with real order execution capabilities.
    Key differences from paper trading:
    - Orders are sent to real exchange
    - Real money position tracking
    - Enhanced safety checks
    """
    
    def __init__(self, config):
        """Initialize live trading bot."""
        super().__init__(config)
        
        # Additional live trading safety checks
        if config.is_paper_mode():
            raise ValueError("Live trading bot requires live mode configuration")
        
        # Override for live trading
        self.is_live_trading = True
        
        self.logger.warning(
            "🚨 LIVE TRADING MODE - REAL MONEY AT RISK 🚨",
            exchange=config.exchange,
            symbols=config.symbols,
            max_loss_usd=f"${config.risk.max_drawdown_pct * 0.01 * 29.68:.2f}"
        )
    
    async def start(self) -> None:
        """Start live trading with additional safety checks."""
        self.logger.warning(
            "🚨 STARTING LIVE TRADING - REAL MONEY AT RISK 🚨",
            exchange=self.config.exchange,
            symbols=self.config.symbols,
            max_loss=f"${self.config.risk.max_drawdown_pct * 0.01 * 29.68:.2f}"
        )
        
        # Use the parent PaperTradingBot's start method
        # but with live trading enabled
        await super().start()
    
    async def _initialize_components(self) -> None:
        """Initialize components with live trading enabled."""
        # Call parent initialization
        await super()._initialize_components()
        
        # Verify that OrderManager has exchange adapter for live trading
        if self.order_manager and self.exchange_adapter:
            # Ensure OrderManager is configured for live trading
            self.order_manager.exchange_adapter = self.exchange_adapter
            
            self.logger.warning(
                "✅ LIVE ORDER EXECUTION ENABLED",
                exchange_adapter=type(self.exchange_adapter).__name__,
                order_manager_has_adapter=self.order_manager.exchange_adapter is not None
            )
        else:
            raise RuntimeError("Failed to initialize live trading components")
