"""
Base exchange adapter interface and common functionality.

This module defines the abstract interface that all exchange adapters must implement,
providing a unified API for market data, order management, and account operations
across different cryptocurrency exchanges.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum

from mm_bot.config import ExchangeConfig, FeeConfig
from mm_bot.logging import LoggerMixin
from mm_bot.utils.latency import LatencyTracker


class ExchangeStatus(Enum):
    """Exchange connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class MarketInfo:
    """Market information for a trading pair."""
    symbol: str
    base: str
    quote: str
    active: bool
    type: str  # spot, future, swap, etc.
    
    # Trading constraints
    min_amount: float
    max_amount: Optional[float]
    min_price: float
    max_price: Optional[float]
    min_notional: float
    
    # Precision
    amount_precision: int
    price_precision: int
    
    # Fees
    maker_fee: float
    taker_fee: float
    
    # Size increments
    tick_size: float  # Minimum price increment
    lot_size: float   # Minimum amount increment
    
    # Additional metadata
    contract_size: Optional[float] = None  # For futures
    settlement_currency: Optional[str] = None
    margin_enabled: bool = False


@dataclass
class Balance:
    """Account balance information."""
    currency: str
    free: float
    used: float
    total: float


@dataclass
class Position:
    """Position information (for margin/futures)."""
    symbol: str
    side: str  # long, short
    size: float
    entry_price: float
    unrealized_pnl: float
    contracts: Optional[float] = None
    notional: Optional[float] = None
    mark_price: Optional[float] = None
    percentage: Optional[float] = None
    margin_ratio: Optional[float] = None


@dataclass
class OrderBookLevel:
    """Order book level."""
    price: float
    size: float


@dataclass
class OrderBook:
    """Order book snapshot."""
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: int
    nonce: Optional[int] = None


@dataclass
class Trade:
    """Trade information."""
    id: str
    symbol: str
    side: str
    amount: float
    price: float
    cost: float
    fee: float
    timestamp: int
    order_id: Optional[str] = None


@dataclass
class OrderStatus:
    """Order status information."""
    id: str
    client_order_id: Optional[str]
    symbol: str
    side: str
    type: str
    amount: float
    price: Optional[float]
    filled: float
    remaining: float
    status: str
    timestamp: int
    fee: float = 0.0
    average_price: Optional[float] = None
    trades: List[Trade] = None


class BaseExchangeAdapter(ABC, LoggerMixin):
    """
    Abstract base class for exchange adapters.
    
    All exchange adapters must implement this interface to provide
    unified access to different cryptocurrency exchanges.
    """
    
    def __init__(self, config: ExchangeConfig, fee_config: FeeConfig):
        """
        Initialize exchange adapter.
        
        Args:
            config: Exchange configuration
            fee_config: Fee configuration
        """
        super().__init__()
        self.config = config
        self.fee_config = fee_config
        self.name = config.name
        
        # Connection state
        self.status = ExchangeStatus.DISCONNECTED
        self.last_heartbeat = 0
        self.connection_count = 0
        
        # Market data
        self.markets: Dict[str, MarketInfo] = {}
        self.symbols: List[str] = []
        
        # Performance tracking
        self.latency_tracker = LatencyTracker()
        
        # Callbacks
        self.order_book_callbacks: List[Callable[[OrderBook], None]] = []
        self.trade_callbacks: List[Callable[[Trade], None]] = []
        self.order_update_callbacks: List[Callable[[OrderStatus], None]] = []
        
        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        
        self.logger.info(
            "Initialized exchange adapter",
            exchange=self.name,
            sandbox=config.sandbox
        )
    
    # Abstract methods that must be implemented by concrete adapters
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the exchange.
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        pass
    
    @abstractmethod
    async def load_markets(self) -> Dict[str, MarketInfo]:
        """
        Load market information.
        
        Returns:
            Dictionary of symbol -> MarketInfo
        """
        pass
    
    @abstractmethod
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> OrderBook:
        """
        Fetch order book snapshot.
        
        Args:
            symbol: Trading symbol
            limit: Number of levels to fetch
            
        Returns:
            OrderBook snapshot
        """
        pass
    
    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create an order.
        
        Args:
            symbol: Trading symbol
            type: Order type (limit, market, etc.)
            side: Order side (buy, sell)
            amount: Order amount
            price: Order price (for limit orders)
            params: Additional parameters
            
        Returns:
            Order creation response
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol
            
        Returns:
            Cancellation response
        """
        pass
    
    @abstractmethod
    async def fetch_order(self, order_id: str, symbol: str) -> OrderStatus:
        """
        Fetch order status.
        
        Args:
            order_id: Order ID
            symbol: Trading symbol
            
        Returns:
            Order status
        """
        pass
    
    @abstractmethod
    async def fetch_balance(self) -> Dict[str, Balance]:
        """
        Fetch account balance.
        
        Returns:
            Dictionary of currency -> Balance
        """
        pass
    
    @abstractmethod
    async def fetch_positions(self) -> List[Position]:
        """
        Fetch positions (for margin/futures trading).
        
        Returns:
            List of positions
        """
        pass
    
    @abstractmethod
    async def watch_order_book(self, symbol: str, callback: Callable[[OrderBook], None]) -> None:
        """
        Watch order book updates via WebSocket.
        
        Args:
            symbol: Trading symbol
            callback: Callback function for updates
        """
        pass
    
    @abstractmethod
    async def watch_trades(self, symbol: str, callback: Callable[[Trade], None]) -> None:
        """
        Watch trade updates via WebSocket.
        
        Args:
            symbol: Trading symbol
            callback: Callback function for updates
        """
        pass
    
    @abstractmethod
    async def watch_orders(self, callback: Callable[[OrderStatus], None]) -> None:
        """
        Watch order updates via WebSocket.
        
        Args:
            callback: Callback function for updates
        """
        pass
    
    # Common utility methods
    
    def get_market_info(self, symbol: str) -> Optional[MarketInfo]:
        """Get market information for a symbol."""
        return self.markets.get(symbol)
    
    def get_tick_size(self, symbol: str) -> float:
        """Get tick size for a symbol."""
        market = self.get_market_info(symbol)
        return market.tick_size if market else 0.01
    
    def get_lot_size(self, symbol: str) -> float:
        """Get lot size for a symbol."""
        market = self.get_market_info(symbol)
        return market.lot_size if market else 0.001
    
    def get_min_notional(self, symbol: str) -> float:
        """Get minimum notional for a symbol."""
        market = self.get_market_info(symbol)
        return market.min_notional if market else 10.0
    
    def get_maker_fee(self, symbol: str) -> float:
        """Get maker fee for a symbol."""
        market = self.get_market_info(symbol)
        if market:
            return market.maker_fee
        return self.fee_config.maker_bps / 10000.0
    
    def get_taker_fee(self, symbol: str) -> float:
        """Get taker fee for a symbol."""
        market = self.get_market_info(symbol)
        if market:
            return market.taker_fee
        return self.fee_config.taker_bps / 10000.0
    
    def round_to_tick_size(self, price: float, symbol: str) -> float:
        """Round price to tick size."""
        tick_size = self.get_tick_size(symbol)
        return round(price / tick_size) * tick_size
    
    def round_to_lot_size(self, amount: float, symbol: str) -> float:
        """Round amount to lot size."""
        lot_size = self.get_lot_size(symbol)
        return round(amount / lot_size) * lot_size
    
    def validate_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate order parameters.
        
        Args:
            symbol: Trading symbol
            side: Order side
            amount: Order amount
            price: Order price
            
        Returns:
            Tuple of (valid, list_of_errors)
        """
        errors = []
        market = self.get_market_info(symbol)
        
        if not market:
            errors.append(f"Unknown symbol: {symbol}")
            return False, errors
        
        if not market.active:
            errors.append(f"Market not active: {symbol}")
        
        # Amount validation
        if amount < market.min_amount:
            errors.append(f"Amount too small: {amount} < {market.min_amount}")
        
        if market.max_amount and amount > market.max_amount:
            errors.append(f"Amount too large: {amount} > {market.max_amount}")
        
        # Price validation
        if price is not None:
            if price < market.min_price:
                errors.append(f"Price too low: {price} < {market.min_price}")
            
            if market.max_price and price > market.max_price:
                errors.append(f"Price too high: {price} > {market.max_price}")
            
            # Notional validation
            notional = amount * price
            if notional < market.min_notional:
                errors.append(f"Notional too small: {notional} < {market.min_notional}")
        
        return len(errors) == 0, errors
    
    def add_order_book_callback(self, callback: Callable[[OrderBook], None]) -> None:
        """Add order book update callback."""
        self.order_book_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable[[Trade], None]) -> None:
        """Add trade update callback."""
        self.trade_callbacks.append(callback)
    
    def add_order_update_callback(self, callback: Callable[[OrderStatus], None]) -> None:
        """Add order update callback."""
        self.order_update_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get adapter status information."""
        return {
            "exchange": self.name,
            "status": self.status.value,
            "connection_count": self.connection_count,
            "last_heartbeat": self.last_heartbeat,
            "markets_loaded": len(self.markets),
            "symbols": len(self.symbols),
            "latency_stats": self.latency_tracker.get_stats(),
            "request_count": self.request_count,
        }
    
    async def _rate_limit(self) -> None:
        """Apply rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        min_interval = 1.0 / self.config.rate_limit
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _handle_order_book_update(self, order_book: OrderBook) -> None:
        """Handle order book update and call callbacks."""
        for callback in self.order_book_callbacks:
            try:
                callback(order_book)
            except Exception as e:
                self.logger.error("Order book callback error", error=str(e))
    
    def _handle_trade_update(self, trade: Trade) -> None:
        """Handle trade update and call callbacks."""
        for callback in self.trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                self.logger.error("Trade callback error", error=str(e))
    
    def _handle_order_update(self, order_status: OrderStatus) -> None:
        """Handle order update and call callbacks."""
        for callback in self.order_update_callbacks:
            try:
                callback(order_status)
            except Exception as e:
                self.logger.error("Order update callback error", error=str(e))
