"""
Realistic matching engine for backtesting with fees and latency simulation.

This module provides a comprehensive matching engine that simulates
real exchange behavior for accurate backtesting results.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
import random

from mm_bot.execution.order_manager import Order, OrderSide, OrderType, OrderStatus as OrderState
from mm_bot.logging import LoggerMixin
from mm_bot.utils.latency import LatencySimulator
from mm_bot.utils.fees import FeeCalculator


class FillType(Enum):
    """Types of order fills."""
    MAKER = "maker"
    TAKER = "taker"
    PARTIAL = "partial"


@dataclass
class Fill:
    """Order fill information."""
    order_id: str
    symbol: str
    side: str
    size: float
    price: float
    fee: float
    fill_type: FillType
    timestamp: int
    trade_id: str = field(default_factory=lambda: str(int(time.time() * 1000000)))


@dataclass
class MarketState:
    """Current market state for matching."""
    symbol: str
    best_bid: float = 0.0
    best_ask: float = 0.0
    last_trade_price: float = 0.0
    volume_24h: float = 0.0
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


class MatchingEngine(LoggerMixin):
    """
    Realistic matching engine for backtesting.
    
    Features:
    - Price-time priority matching
    - Partial fills and order book impact
    - Realistic latency simulation
    - Fee calculation (maker/taker)
    - Market impact modeling
    - Slippage simulation
    - Order rejection scenarios
    """
    
    def __init__(
        self,
        fee_calculator: FeeCalculator,
        latency_simulator: Optional[LatencySimulator] = None,
        slippage_model: Optional[Callable] = None
    ):
        """
        Initialize matching engine.
        
        Args:
            fee_calculator: Fee calculation component
            latency_simulator: Latency simulation component
            slippage_model: Market impact/slippage model
        """
        super().__init__()
        self.fee_calculator = fee_calculator
        self.latency_simulator = latency_simulator or LatencySimulator()
        self.slippage_model = slippage_model or self._default_slippage_model
        
        # Order books (symbol -> side -> price -> orders)
        self.order_books: Dict[str, Dict[str, Dict[float, List[Order]]]] = defaultdict(
            lambda: {"buy": defaultdict(list), "sell": defaultdict(list)}
        )
        
        # Market state tracking
        self.market_states: Dict[str, MarketState] = {}
        
        # Fill tracking
        self.fills: List[Fill] = []
        self.fill_callbacks: List[Callable[[Fill], None]] = []
        
        # Statistics
        self.total_orders = 0
        self.total_fills = 0
        self.total_volume = 0.0
        self.total_fees = 0.0
        self.rejected_orders = 0
        
        # Configuration
        self.min_fill_size = 0.0001  # Minimum fill size
        self.max_fill_ratio = 0.8    # Maximum % of order that can fill immediately
        self.rejection_rate = 0.001  # Random rejection rate for realism
        
        self.logger.info("Initialized matching engine")
    
    def submit_order(self, order: Order) -> bool:
        """
        Submit order to matching engine.
        
        Args:
            order: Order to submit
            
        Returns:
            True if order was accepted
        """
        self.total_orders += 1
        
        # Simulate latency
        processing_delay = self.latency_simulator.get_order_latency()
        if processing_delay > 0:
            # In a real implementation, this would be handled asynchronously
            pass
        
        # Random rejection for realism
        if random.random() < self.rejection_rate:
            self.rejected_orders += 1
            order.status = OrderState.REJECTED
            order.last_error = "Random rejection (simulated)"
            self.logger.debug("Order rejected randomly", order_id=order.id)
            return False
        
        # Validate order
        if not self._validate_order(order):
            self.rejected_orders += 1
            return False
        
        # Update order status
        order.status = OrderState.OPEN
        order.submitted_at = int(time.time() * 1000)
        
        # Try to match immediately
        fills = self._match_order(order)
        
        # Process fills
        for fill in fills:
            self._process_fill(fill)
        
        # Add remaining order to book if not fully filled
        if order.remaining_amount > self.min_fill_size:
            self._add_to_order_book(order)
        
        return True
    
    def cancel_order(self, order: Order) -> bool:
        """Cancel an order."""
        if order.status != OrderState.OPEN:
            return False
        
        # Remove from order book
        self._remove_from_order_book(order)
        
        # Update status
        order.status = OrderState.CANCELLED
        order.updated_at = int(time.time() * 1000)
        
        self.logger.debug("Order cancelled", order_id=order.id)
        return True
    
    def update_market_data(
        self,
        symbol: str,
        best_bid: float,
        best_ask: float,
        last_trade_price: Optional[float] = None
    ) -> None:
        """Update market data for matching decisions."""
        if symbol not in self.market_states:
            self.market_states[symbol] = MarketState(symbol)
        
        market_state = self.market_states[symbol]
        market_state.best_bid = best_bid
        market_state.best_ask = best_ask
        if last_trade_price:
            market_state.last_trade_price = last_trade_price
        market_state.timestamp = int(time.time() * 1000)
        
        # Check for fills against existing orders
        self._check_existing_orders(symbol)
    
    def get_order_book(self, symbol: str, depth: int = 10) -> Dict[str, List[Tuple[float, float]]]:
        """Get current order book state."""
        if symbol not in self.order_books:
            return {"bids": [], "asks": []}
        
        book = self.order_books[symbol]
        
        # Sort and aggregate bids (descending price)
        bid_prices = sorted(book["buy"].keys(), reverse=True)[:depth]
        bids = []
        for price in bid_prices:
            total_size = sum(order.remaining_amount for order in book["buy"][price])
            if total_size > 0:
                bids.append((price, total_size))
        
        # Sort and aggregate asks (ascending price)
        ask_prices = sorted(book["sell"].keys())[:depth]
        asks = []
        for price in ask_prices:
            total_size = sum(order.remaining_amount for order in book["sell"][price])
            if total_size > 0:
                asks.append((price, total_size))
        
        return {"bids": bids, "asks": asks}
    
    def add_fill_callback(self, callback: Callable[[Fill], None]) -> None:
        """Add callback for order fills."""
        self.fill_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, any]:
        """Get matching engine statistics."""
        return {
            "total_orders": self.total_orders,
            "total_fills": self.total_fills,
            "total_volume": self.total_volume,
            "total_fees": self.total_fees,
            "rejected_orders": self.rejected_orders,
            "rejection_rate": self.rejected_orders / max(self.total_orders, 1),
            "fill_rate": self.total_fills / max(self.total_orders, 1),
            "average_fill_size": self.total_volume / max(self.total_fills, 1),
            "symbols": list(self.market_states.keys()),
            "active_orders": sum(
                len(orders) 
                for symbol_book in self.order_books.values()
                for side_book in symbol_book.values()
                for orders in side_book.values()
            )
        }
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order before matching."""
        # Basic validation
        if order.amount <= 0:
            order.status = OrderState.REJECTED
            order.last_error = "Invalid amount"
            return False
        
        if order.type == OrderType.LIMIT and (order.price is None or order.price <= 0):
            order.status = OrderState.REJECTED
            order.last_error = "Invalid price for limit order"
            return False
        
        # Market state validation
        market_state = self.market_states.get(order.symbol)
        if not market_state:
            order.status = OrderState.REJECTED
            order.last_error = "No market data available"
            return False
        
        return True
    
    def _match_order(self, order: Order) -> List[Fill]:
        """Match order against existing order book."""
        fills = []
        
        if order.type == OrderType.MARKET:
            fills.extend(self._match_market_order(order))
        elif order.type == OrderType.LIMIT:
            fills.extend(self._match_limit_order(order))
        
        return fills
    
    def _match_market_order(self, order: Order) -> List[Fill]:
        """Match market order with slippage and impact."""
        fills = []
        market_state = self.market_states.get(order.symbol)
        if not market_state:
            return fills
        
        # Calculate effective price with slippage
        base_price = market_state.best_ask if order.is_buy else market_state.best_bid
        if base_price <= 0:
            return fills
        
        # Apply slippage model
        slippage = self.slippage_model(order.symbol, order.amount, base_price)
        effective_price = base_price * (1 + slippage if order.is_buy else 1 - slippage)
        
        # Limit fill amount for realism
        max_fill_amount = order.amount * self.max_fill_ratio
        fill_amount = min(order.remaining_amount, max_fill_amount)
        
        if fill_amount >= self.min_fill_size:
            # Calculate fee
            fee = self.fee_calculator.calculate_taker_fee(
                order.symbol, fill_amount * effective_price
            )
            
            # Create fill
            fill = Fill(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side.value,
                size=fill_amount,
                price=effective_price,
                fee=fee,
                fill_type=FillType.TAKER,
                timestamp=int(time.time() * 1000)
            )
            
            fills.append(fill)
            
            # Update order
            order.update_fill(fill_amount, effective_price, fee)
        
        return fills
    
    def _match_limit_order(self, order: Order) -> List[Fill]:
        """Match limit order against market prices."""
        fills = []
        market_state = self.market_states.get(order.symbol)
        if not market_state:
            return fills
        
        # Check if limit order can fill immediately
        can_fill = False
        market_price = 0.0
        
        if order.is_buy and market_state.best_ask > 0:
            if order.price >= market_state.best_ask:
                can_fill = True
                market_price = market_state.best_ask
        elif order.is_sell and market_state.best_bid > 0:
            if order.price <= market_state.best_bid:
                can_fill = True
                market_price = market_state.best_bid
        
        if can_fill:
            # Partial fill for realism
            fill_ratio = min(self.max_fill_ratio, random.uniform(0.3, 1.0))
            fill_amount = min(order.remaining_amount, order.amount * fill_ratio)
            
            if fill_amount >= self.min_fill_size:
                # Use order price (price improvement)
                fill_price = order.price
                
                # Calculate fee (taker since it filled immediately)
                fee = self.fee_calculator.calculate_taker_fee(
                    order.symbol, fill_amount * fill_price
                )
                
                fill = Fill(
                    order_id=order.id,
                    symbol=order.symbol,
                    side=order.side.value,
                    size=fill_amount,
                    price=fill_price,
                    fee=fee,
                    fill_type=FillType.TAKER,
                    timestamp=int(time.time() * 1000)
                )
                
                fills.append(fill)
                order.update_fill(fill_amount, fill_price, fee)
        
        return fills
    
    def _add_to_order_book(self, order: Order) -> None:
        """Add order to order book."""
        if order.remaining_amount <= self.min_fill_size:
            return
        
        side = "buy" if order.is_buy else "sell"
        price = order.price
        
        if price is None:
            return
        
        self.order_books[order.symbol][side][price].append(order)
        
        self.logger.debug(
            "Added order to book",
            order_id=order.id,
            symbol=order.symbol,
            side=side,
            price=price,
            size=order.remaining_amount
        )
    
    def _remove_from_order_book(self, order: Order) -> None:
        """Remove order from order book."""
        side = "buy" if order.is_buy else "sell"
        price = order.price
        
        if price is None or order.symbol not in self.order_books:
            return
        
        book = self.order_books[order.symbol][side]
        if price in book and order in book[price]:
            book[price].remove(order)
            
            # Clean up empty price levels
            if not book[price]:
                del book[price]
    
    def _check_existing_orders(self, symbol: str) -> None:
        """Check existing orders for potential fills."""
        if symbol not in self.order_books:
            return
        
        market_state = self.market_states[symbol]
        book = self.order_books[symbol]
        
        # Check buy orders against best ask
        if market_state.best_ask > 0:
            for price in list(book["buy"].keys()):
                if price >= market_state.best_ask:
                    orders = list(book["buy"][price])
                    for order in orders:
                        if random.random() < 0.1:  # 10% chance of fill
                            self._fill_maker_order(order, market_state.best_ask)
        
        # Check sell orders against best bid
        if market_state.best_bid > 0:
            for price in list(book["sell"].keys()):
                if price <= market_state.best_bid:
                    orders = list(book["sell"][price])
                    for order in orders:
                        if random.random() < 0.1:  # 10% chance of fill
                            self._fill_maker_order(order, market_state.best_bid)
    
    def _fill_maker_order(self, order: Order, market_price: float) -> None:
        """Fill a maker order at market price."""
        # Random partial fill
        fill_ratio = random.uniform(0.1, 1.0)
        fill_amount = min(order.remaining_amount, order.amount * fill_ratio)
        
        if fill_amount < self.min_fill_size:
            return
        
        # Use order price (maker gets price improvement)
        fill_price = order.price
        
        # Calculate maker fee
        fee = self.fee_calculator.calculate_maker_fee(
            order.symbol, fill_amount * fill_price
        )
        
        fill = Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side.value,
            size=fill_amount,
            price=fill_price,
            fee=fee,
            fill_type=FillType.MAKER,
            timestamp=int(time.time() * 1000)
        )
        
        # Process fill
        self._process_fill(fill)
        order.update_fill(fill_amount, fill_price, fee)
        
        # Remove from book if fully filled
        if order.remaining_amount <= self.min_fill_size:
            self._remove_from_order_book(order)
    
    def _process_fill(self, fill: Fill) -> None:
        """Process a fill and update statistics."""
        self.fills.append(fill)
        self.total_fills += 1
        self.total_volume += fill.size * fill.price
        self.total_fees += fill.fee
        
        # Call callbacks
        for callback in self.fill_callbacks:
            try:
                callback(fill)
            except Exception as e:
                self.logger.error("Fill callback error", error=str(e))
        
        self.logger.debug(
            "Order filled",
            order_id=fill.order_id,
            symbol=fill.symbol,
            side=fill.side,
            size=fill.size,
            price=fill.price,
            fee=fill.fee,
            fill_type=fill.fill_type.value
        )
    
    def _default_slippage_model(self, symbol: str, amount: float, price: float) -> float:
        """Default slippage model based on order size."""
        notional = amount * price
        
        # Simple square root model
        if notional < 1000:
            return 0.0001  # 1 bps
        elif notional < 10000:
            return 0.0005  # 5 bps
        elif notional < 100000:
            return 0.001   # 10 bps
        else:
            return 0.002   # 20 bps
