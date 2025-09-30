"""
Async order management with throttling, post-only logic, and risk controls.

This module provides comprehensive order management capabilities including:
- Async order placement and cancellation
- Rate limiting and throttling
- Order state tracking and lifecycle management
- Post-only order handling
- Idempotency and retry logic
- Risk checks and position tracking
"""

import asyncio
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any

from mm_bot.config import ExecutionConfig
from mm_bot.logging import LoggerMixin
from mm_bot.utils.time import timestamp_ms, RateLimiter
from mm_bot.utils.latency import LatencyTracker


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"          # Order created but not sent
    SUBMITTED = "submitted"      # Order sent to exchange
    OPEN = "open"               # Order accepted by exchange
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"           # Internal failure


class TimeInForce(Enum):
    """Time in force enumeration."""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTD = "GTD"  # Good Till Date


@dataclass
class Order:
    """Order representation with full lifecycle tracking."""
    
    # Core order fields
    id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    type: OrderType
    amount: float
    price: Optional[float] = None
    
    # Order parameters
    time_in_force: TimeInForce = TimeInForce.GTC
    post_only: bool = True
    reduce_only: bool = False
    
    # Status and tracking
    status: OrderStatus = OrderStatus.PENDING
    exchange_order_id: Optional[str] = None
    
    # Execution details
    filled_amount: float = 0.0
    remaining_amount: float = field(init=False)
    average_price: Optional[float] = None
    fee: float = 0.0
    
    # Timestamps
    created_at: int = field(default_factory=timestamp_ms)
    submitted_at: Optional[int] = None
    updated_at: Optional[int] = None
    
    # Retry and error tracking
    retry_count: int = 0
    last_error: Optional[str] = None
    
    # Metadata
    strategy_id: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize computed fields."""
        self.remaining_amount = self.amount
    
    @property
    def is_buy(self) -> bool:
        """Check if order is a buy order."""
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        """Check if order is a sell order."""
        return self.side == OrderSide.SELL
    
    @property
    def is_active(self) -> bool:
        """Check if order is active (can be filled or cancelled)."""
        return self.status in {OrderStatus.SUBMITTED, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED}
    
    @property
    def is_terminal(self) -> bool:
        """Check if order is in terminal state."""
        return self.status in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.FAILED}
    
    @property
    def fill_ratio(self) -> float:
        """Get fill ratio (0.0 to 1.0)."""
        return self.filled_amount / self.amount if self.amount > 0 else 0.0
    
    def update_fill(self, filled_amount: float, price: float, fee: float = 0.0) -> None:
        """Update order with fill information."""
        self.filled_amount += filled_amount
        self.remaining_amount = max(0.0, self.amount - self.filled_amount)
        self.fee += fee
        self.updated_at = timestamp_ms()
        
        # Update average price
        if self.average_price is None:
            self.average_price = price
        else:
            # Volume-weighted average price
            total_filled = self.filled_amount
            if total_filled > 0:
                self.average_price = ((self.average_price * (total_filled - filled_amount)) + (price * filled_amount)) / total_filled
        
        # Update status
        if self.remaining_amount <= 1e-8:  # Essentially zero
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary for logging/serialization."""
        return {
            "id": self.id,
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.type.value,
            "amount": self.amount,
            "price": self.price,
            "status": self.status.value,
            "filled_amount": self.filled_amount,
            "remaining_amount": self.remaining_amount,
            "average_price": self.average_price,
            "fee": self.fee,
            "created_at": self.created_at,
            "submitted_at": self.submitted_at,
            "updated_at": self.updated_at,
            "retry_count": self.retry_count,
            "last_error": self.last_error,
        }


class OrderManager(LoggerMixin):
    """
    Async order manager with comprehensive order lifecycle management.
    
    Features:
    - Async order placement and cancellation
    - Rate limiting and throttling
    - Order state tracking
    - Post-only order handling
    - Retry logic with exponential backoff
    - Risk checks and position tracking
    - Idempotency support
    """
    
    def __init__(self, config: ExecutionConfig, exchange_adapter=None, risk_manager=None):
        """
        Initialize order manager.
        
        Args:
            config: Execution configuration
            exchange_adapter: Exchange adapter for order operations
            risk_manager: Risk manager for order validation
        """
        super().__init__()
        self.config = config
        self.exchange_adapter = exchange_adapter
        self.risk_manager = risk_manager
        
        # Order tracking
        self.orders: Dict[str, Order] = {}  # order_id -> Order
        self.orders_by_client_id: Dict[str, Order] = {}  # client_order_id -> Order
        self.orders_by_exchange_id: Dict[str, Order] = {}  # exchange_order_id -> Order
        self.active_orders: Set[str] = set()  # Set of active order IDs
        self.orders_by_symbol: Dict[str, Set[str]] = defaultdict(set)
        
        # Rate limiting
        self.rate_limiter = RateLimiter(60.0 / config.max_order_retries if hasattr(config, 'max_order_retries') else 10.0)
        self.last_operation_time = 0
        
        # Performance tracking
        self.latency_tracker = LatencyTracker()
        self.order_stats = {
            "orders_placed": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "orders_rejected": 0,
            "total_volume": 0.0,
            "total_fees": 0.0,
        }
        
        # Callbacks
        self.fill_callbacks: List[Callable[[Order], None]] = []
        self.cancel_callbacks: List[Callable[[Order], None]] = []
        self.reject_callbacks: List[Callable[[Order, str], None]] = []
        
        # Background tasks
        self._running = False
        self._background_tasks: Set[asyncio.Task] = set()
        
        self.logger.info(
            "Initialized order manager",
            post_only=config.post_only,
            throttle_ms=config.throttle_ms,
            max_retries=config.max_order_retries
        )
    
    async def start(self) -> None:
        """Start the order manager background tasks."""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        task = asyncio.create_task(self._order_lifecycle_monitor())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        
        self.logger.info("Order manager started")
    
    async def stop(self) -> None:
        """Stop the order manager and cancel all background tasks."""
        self._running = False
        
        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Cancel all active orders
        await self.cancel_all_orders("shutdown")
        
        self.logger.info("Order manager stopped")
    
    def add_fill_callback(self, callback: Callable[[Order], None]) -> None:
        """Add callback for order fills."""
        self.fill_callbacks.append(callback)
    
    def add_cancel_callback(self, callback: Callable[[Order], None]) -> None:
        """Add callback for order cancellations."""
        self.cancel_callbacks.append(callback)
    
    def add_reject_callback(self, callback: Callable[[Order, str], None]) -> None:
        """Add callback for order rejections."""
        self.reject_callbacks.append(callback)
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.LIMIT,
        time_in_force: TimeInForce = TimeInForce.GTC,
        client_order_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> Optional[Order]:
        """
        Place a new order.
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            amount: Order amount
            price: Order price (required for limit orders)
            order_type: Order type
            time_in_force: Time in force
            client_order_id: Custom client order ID
            strategy_id: Strategy identifier
            tags: Additional metadata
            
        Returns:
            Order object if successful, None otherwise
        """
        # Generate IDs
        order_id = str(uuid.uuid4())
        if client_order_id is None:
            client_order_id = f"mm_{int(time.time() * 1000)}_{order_id[:8]}"
        
        # Create order
        order = Order(
            id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            type=order_type,
            amount=amount,
            price=price,
            time_in_force=time_in_force,
            post_only=self.config.post_only,
            strategy_id=strategy_id,
            tags=tags or {}
        )
        
        # Validate order
        if not self._validate_order(order):
            return None
        
        # Risk validation
        if self.risk_manager:
            allowed, violations = self.risk_manager.check_order_allowed(
                symbol, side.value, amount, price
            )
            if not allowed:
                order.status = OrderStatus.REJECTED
                order.last_error = f"Risk limits violated: {', '.join(violations)}"
                self.logger.warning(
                    "Order rejected by risk manager",
                    order_id=order.id,
                    violations=violations
                )
                return None
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        # Apply throttling
        current_time = time.time()
        time_since_last = (current_time - self.last_operation_time) * 1000
        if time_since_last < self.config.throttle_ms:
            sleep_time = (self.config.throttle_ms - time_since_last) / 1000
            await asyncio.sleep(sleep_time)
        
        # Store order
        self.orders[order.id] = order
        self.orders_by_client_id[order.client_order_id] = order
        self.orders_by_symbol[symbol].add(order.id)
        
        # Submit order
        success = await self._submit_order(order)
        
        self.last_operation_time = time.time()
        
        if success:
            self.active_orders.add(order.id)
            self.order_stats["orders_placed"] += 1
            
            self.logger.info(
                "Order placed",
                order_id=order.id,
                symbol=symbol,
                side=side.value,
                amount=amount,
                price=price,
                client_order_id=client_order_id
            )
            
            return order
        else:
            # Clean up failed order
            self._remove_order_from_tracking(order)
            return None
    
    async def cancel_order(self, order_id: str, reason: str = "manual") -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            reason: Cancellation reason
            
        Returns:
            True if cancellation was successful
        """
        order = self.orders.get(order_id)
        if not order:
            self.logger.warning("Order not found for cancellation", order_id=order_id)
            return False
        
        if not order.is_active:
            self.logger.warning(
                "Order not active for cancellation",
                order_id=order_id,
                status=order.status.value
            )
            return False
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        # Apply throttling
        current_time = time.time()
        time_since_last = (current_time - self.last_operation_time) * 1000
        if time_since_last < self.config.throttle_ms:
            sleep_time = (self.config.throttle_ms - time_since_last) / 1000
            await asyncio.sleep(sleep_time)
        
        # Cancel order
        success = await self._cancel_order(order, reason)
        
        self.last_operation_time = time.time()
        
        if success:
            self.order_stats["orders_cancelled"] += 1
            
            self.logger.info(
                "Order cancelled",
                order_id=order_id,
                reason=reason,
                symbol=order.symbol
            )
        
        return success
    
    async def cancel_all_orders(self, reason: str = "cancel_all") -> int:
        """
        Cancel all active orders.
        
        Args:
            reason: Cancellation reason
            
        Returns:
            Number of orders cancelled
        """
        active_order_ids = list(self.active_orders)
        cancelled_count = 0
        
        # Cancel orders in parallel
        tasks = []
        for order_id in active_order_ids:
            task = asyncio.create_task(self.cancel_order(order_id, reason))
            tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            cancelled_count = sum(1 for result in results if result is True)
        
        self.logger.info(
            "Cancelled all orders",
            total_orders=len(active_order_ids),
            cancelled_count=cancelled_count,
            reason=reason
        )
        
        return cancelled_count
    
    async def cancel_orders_by_symbol(self, symbol: str, reason: str = "symbol_cancel") -> int:
        """Cancel all orders for a specific symbol."""
        order_ids = list(self.orders_by_symbol[symbol] & self.active_orders)
        cancelled_count = 0
        
        for order_id in order_ids:
            if await self.cancel_order(order_id, reason):
                cancelled_count += 1
        
        return cancelled_count
    
    async def replace_order(
        self,
        order_id: str,
        new_amount: Optional[float] = None,
        new_price: Optional[float] = None
    ) -> Optional[Order]:
        """
        Replace an existing order with new parameters.
        
        Args:
            order_id: Order ID to replace
            new_amount: New order amount
            new_price: New order price
            
        Returns:
            New order if successful
        """
        old_order = self.orders.get(order_id)
        if not old_order or not old_order.is_active:
            return None
        
        # Cancel old order
        if not await self.cancel_order(order_id, "replace"):
            return None
        
        # Place new order
        new_order = await self.place_order(
            symbol=old_order.symbol,
            side=old_order.side,
            amount=new_amount or old_order.amount,
            price=new_price or old_order.price,
            order_type=old_order.type,
            time_in_force=old_order.time_in_force,
            strategy_id=old_order.strategy_id,
            tags=old_order.tags
        )
        
        return new_order
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_order_by_client_id(self, client_order_id: str) -> Optional[Order]:
        """Get order by client ID."""
        return self.orders_by_client_id.get(client_order_id)
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active orders, optionally filtered by symbol."""
        if symbol:
            order_ids = self.orders_by_symbol[symbol] & self.active_orders
        else:
            order_ids = self.active_orders
        
        return [self.orders[order_id] for order_id in order_ids if order_id in self.orders]
    
    def get_position_from_orders(self, symbol: str) -> float:
        """Calculate net position from active orders."""
        position = 0.0
        
        for order_id in self.orders_by_symbol[symbol]:
            order = self.orders.get(order_id)
            if order and order.is_active:
                if order.is_buy:
                    position += order.remaining_amount
                else:
                    position -= order.remaining_amount
        
        return position
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get order manager statistics."""
        active_count = len(self.active_orders)
        total_count = len(self.orders)
        
        stats = {
            "active_orders": active_count,
            "total_orders": total_count,
            "orders_by_status": defaultdict(int),
            "latency_stats": self.latency_tracker.get_stats(),
            **self.order_stats
        }
        
        # Count orders by status
        for order in self.orders.values():
            stats["orders_by_status"][order.status.value] += 1
        
        return stats
    
    async def _submit_order(self, order: Order) -> bool:
        """Submit order to exchange."""
        if not self.exchange_adapter:
            # Simulation mode - just mark as submitted
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = timestamp_ms()
            return True
        
        try:
            # Track latency
            operation_id = f"submit_{order.id}"
            self.latency_tracker.start_timer(operation_id)
            
            # Submit to exchange
            result = await self.exchange_adapter.create_order(
                symbol=order.symbol,
                type=order.type.value,
                side=order.side.value,
                amount=order.amount,
                price=order.price,
                params={
                    "timeInForce": order.time_in_force.value,
                    "postOnly": order.post_only,
                    "clientOrderId": order.client_order_id,
                }
            )
            
            # Record latency
            latency = self.latency_tracker.end_timer(operation_id)
            
            if result and result.get("id"):
                order.exchange_order_id = result["id"]
                order.status = OrderStatus.OPEN
                order.submitted_at = timestamp_ms()
                
                # Store exchange ID mapping
                self.orders_by_exchange_id[order.exchange_order_id] = order
                
                return True
            else:
                order.status = OrderStatus.REJECTED
                order.last_error = "Exchange rejected order"
                self._handle_order_rejection(order, "Exchange rejection")
                return False
                
        except Exception as e:
            order.status = OrderStatus.FAILED
            order.last_error = str(e)
            
            self.logger.error(
                "Order submission failed",
                order_id=order.id,
                error=str(e),
                retry_count=order.retry_count
            )
            
            # Retry logic
            if order.retry_count < self.config.max_order_retries:
                order.retry_count += 1
                await asyncio.sleep(min(2 ** order.retry_count, 30))  # Exponential backoff
                return await self._submit_order(order)
            else:
                self._handle_order_rejection(order, f"Max retries exceeded: {e}")
                return False
    
    async def _cancel_order(self, order: Order, reason: str) -> bool:
        """Cancel order on exchange."""
        if not self.exchange_adapter:
            # Simulation mode
            order.status = OrderStatus.CANCELLED
            order.updated_at = timestamp_ms()
            self.active_orders.discard(order.id)
            self._handle_order_cancellation(order, reason)
            return True
        
        try:
            # Track latency
            operation_id = f"cancel_{order.id}"
            self.latency_tracker.start_timer(operation_id)
            
            # Cancel on exchange
            result = await self.exchange_adapter.cancel_order(
                order.exchange_order_id or order.id,
                order.symbol
            )
            
            # Record latency
            latency = self.latency_tracker.end_timer(operation_id)
            
            if result:
                order.status = OrderStatus.CANCELLED
                order.updated_at = timestamp_ms()
                self.active_orders.discard(order.id)
                self._handle_order_cancellation(order, reason)
                return True
            else:
                self.logger.warning(
                    "Order cancellation failed",
                    order_id=order.id,
                    exchange_order_id=order.exchange_order_id
                )
                return False
                
        except Exception as e:
            self.logger.error(
                "Order cancellation error",
                order_id=order.id,
                error=str(e)
            )
            return False
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order before submission."""
        # Basic validation
        if order.amount <= 0:
            order.last_error = "Invalid order amount"
            return False
        
        if order.type == OrderType.LIMIT and (order.price is None or order.price <= 0):
            order.last_error = "Invalid limit order price"
            return False
        
        # Check for duplicate client order ID
        if order.client_order_id in self.orders_by_client_id:
            order.last_error = "Duplicate client order ID"
            return False
        
        return True
    
    def _remove_order_from_tracking(self, order: Order) -> None:
        """Remove order from all tracking structures."""
        self.orders.pop(order.id, None)
        self.orders_by_client_id.pop(order.client_order_id, None)
        if order.exchange_order_id:
            self.orders_by_exchange_id.pop(order.exchange_order_id, None)
        self.active_orders.discard(order.id)
        self.orders_by_symbol[order.symbol].discard(order.id)
    
    def _handle_order_fill(self, order: Order, filled_amount: float, price: float, fee: float = 0.0) -> None:
        """Handle order fill event."""
        order.update_fill(filled_amount, price, fee)
        
        # Update statistics
        self.order_stats["total_volume"] += filled_amount * price
        self.order_stats["total_fees"] += fee
        
        if order.status == OrderStatus.FILLED:
            self.active_orders.discard(order.id)
            self.order_stats["orders_filled"] += 1
        
        # Call callbacks
        for callback in self.fill_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error("Fill callback error", error=str(e))
    
    def _handle_order_cancellation(self, order: Order, reason: str) -> None:
        """Handle order cancellation event."""
        # Call callbacks
        for callback in self.cancel_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error("Cancel callback error", error=str(e))
    
    def _handle_order_rejection(self, order: Order, reason: str) -> None:
        """Handle order rejection event."""
        self.order_stats["orders_rejected"] += 1
        
        # Call callbacks
        for callback in self.reject_callbacks:
            try:
                callback(order, reason)
            except Exception as e:
                self.logger.error("Reject callback error", error=str(e))
    
    async def _order_lifecycle_monitor(self) -> None:
        """Background task to monitor order lifecycles."""
        while self._running:
            try:
                current_time = timestamp_ms()
                
                # Check for orders that need replacement due to age
                orders_to_replace = []
                for order_id in list(self.active_orders):
                    order = self.orders.get(order_id)
                    if not order:
                        continue
                    
                    # Check if order is too old
                    age_ms = current_time - (order.submitted_at or order.created_at)
                    if age_ms > self.config.replace_after_ms:
                        orders_to_replace.append(order_id)
                
                # Replace old orders
                for order_id in orders_to_replace:
                    order = self.orders.get(order_id)
                    if order and order.is_active:
                        self.logger.info(
                            "Replacing aged order",
                            order_id=order_id,
                            age_ms=current_time - (order.submitted_at or order.created_at)
                        )
                        await self.replace_order(order_id)
                
                # Sleep before next check
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error("Order lifecycle monitor error", error=str(e))
                await asyncio.sleep(5.0)
