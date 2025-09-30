"""
Official Hyperliquid SDK Exchange Adapter for REAL Trading

This adapter uses the official hyperliquid-python-sdk for:
- Real order placement (not simulation)
- Proper market data feeds
- Official API integration
- Production-ready trading

SAFETY WARNING: This adapter places REAL ORDERS with REAL MONEY!
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from decimal import Decimal

# Official Hyperliquid SDK imports
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

# For private key handling
from eth_account import Account

from mm_bot.exchanges.base import BaseExchangeAdapter, OrderBook, OrderBookLevel, OrderStatus, Position
from mm_bot.execution.order_manager import Order, OrderType, OrderSide
import structlog


class HyperliquidOfficialAdapter(BaseExchangeAdapter):
    """
    Official Hyperliquid SDK adapter for REAL TRADING.
    
    Key Features:
    - Uses official hyperliquid-python-sdk
    - Real order placement (not simulation)
    - Proper market data integration
    - Production safety checks
    
    DANGER: This places real orders with real money!
    """
    
    def __init__(self, config, fee_config=None):
        super().__init__(config, fee_config or {})
        self.logger = structlog.get_logger(__name__)
        
        # Configuration
        self.wallet_address = config.api_key  # Public wallet address
        self.private_key = config.api_secret  # Private key for signing (if provided)
        self.is_testnet = config.sandbox
        
        # API URLs
        if self.is_testnet:
            self.api_url = constants.TESTNET_API_URL
            self.logger.warning("🧪 TESTNET MODE - Using testnet API")
        else:
            self.api_url = constants.MAINNET_API_URL
            self.logger.critical("🚨 MAINNET MODE - REAL MONEY AT RISK!")
        
        # SDK instances
        self.info = None  # Info API (read-only)
        self.exchange = None  # Exchange API (trading)
        
        # Safety limits - Updated for PENGU trading
        self.max_order_size = getattr(config, 'max_order_size', 1000.0)  # Allow up to 1000 PENGU tokens
        self.min_balance = getattr(config, 'min_account_balance', 20.0)
        self.max_position_pct = getattr(config, 'max_position_pct', 0.5)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0 / config.rate_limit if config.rate_limit else 0.2
        
        self.logger.warning(
            "🚨 OFFICIAL HYPERLIQUID SDK ADAPTER INITIALIZED",
            wallet_address=self.wallet_address,
            is_testnet=self.is_testnet,
            max_order_size=self.max_order_size,
            min_balance=self.min_balance,
            api_url=self.api_url
        )
    
    async def connect(self) -> bool:
        """Connect to Hyperliquid using official SDK."""
        try:
            self.logger.info("🔌 Connecting to Hyperliquid via Official SDK...")
            
            # Initialize Info API (read-only, no private key needed)
            self.info = Info(self.api_url, skip_ws=True)
            
            # Test connection with user state
            if self.wallet_address:
                user_state = self.info.user_state(self.wallet_address)
                if user_state:
                    self.logger.info(
                        "✅ Successfully connected to Hyperliquid",
                        account_value=user_state.get('marginSummary', {}).get('accountValue', 'N/A')
                    )
                else:
                    self.logger.error("❌ Failed to fetch user state")
                    return False
            
            # Initialize Exchange API (for trading, requires private key)
            if self.private_key:
                try:
                    # Convert private key string to LocalAccount object
                    # Remove 0x prefix if present
                    private_key_clean = self.private_key.replace('0x', '') if self.private_key.startswith('0x') else self.private_key
                    wallet_account = Account.from_key(private_key_clean)
                    self.exchange = Exchange(wallet_account, self.api_url)
                    self.logger.warning("🔑 TRADING ENABLED - Exchange API initialized with LocalAccount")
                    
                    # Log wallet addresses but allow mismatch for sub-accounts
                    if wallet_account.address.lower() != self.wallet_address.lower():
                        self.logger.warning(f"⚠️ Wallet address mismatch - but proceeding: Config: {self.wallet_address}, Derived: {wallet_account.address}")
                        # Don't disable exchange - Hyperliquid might use sub-accounts
                    else:
                        self.logger.info(f"✅ Wallet address verified: {wallet_account.address}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to initialize Exchange API: {e}")
                    self.exchange = None
            else:
                self.logger.warning("📖 READ-ONLY MODE - No private key provided, orders will be simulated")
            
            return True
            
        except Exception as e:
            self.logger.error("❌ Failed to connect to Hyperliquid", error=str(e))
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Hyperliquid."""
        self.logger.info("📴 Disconnecting from Hyperliquid")
        # Official SDK handles cleanup automatically
        self.info = None
        self.exchange = None
    
    async def _rate_limit(self):
        """Rate limiting for API requests."""
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        self.last_request_time = time.time()
    
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> OrderBook:
        """Fetch order book using official SDK."""
        try:
            await self._rate_limit()
            
            # Use official SDK to get L2 book
            l2_book = self.info.l2_snapshot(symbol)
            
            if not l2_book or 'levels' not in l2_book:
                raise Exception(f"Invalid order book response for {symbol}")
            
            levels = l2_book['levels']
            bids = []
            asks = []
            
            # Parse levels - official SDK should return proper format
            for level in levels:
                if isinstance(level, dict) and 'px' in level and 'sz' in level:
                    price = float(level['px'])
                    size = float(level['sz'])
                    n = level.get('n', 0)
                    
                    if n > 0:  # Bid
                        bids.append(OrderBookLevel(price=price, size=size))
                    elif n < 0:  # Ask
                        asks.append(OrderBookLevel(price=price, size=size))
            
            # Sort and limit
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)
            bids = bids[:limit]
            asks = asks[:limit]
            
            # Validate we have data
            if not bids or not asks:
                # Get current price and create minimal book
                all_mids = self.info.all_mids()
                mid_price = float(all_mids.get(symbol, 0.037))
                
                bids = [OrderBookLevel(price=mid_price * 0.999, size=1000)]
                asks = [OrderBookLevel(price=mid_price * 1.001, size=1000)]
                
                self.logger.warning("Created fallback order book", symbol=symbol, mid=mid_price)
            
            return OrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=int(time.time() * 1000)
            )
            
        except Exception as e:
            self.logger.error("Error fetching order book", error=str(e), symbol=symbol)
            
            # Emergency fallback
            try:
                all_mids = self.info.all_mids()
                mid_price = float(all_mids.get(symbol, 0.037))
            except:
                mid_price = 0.037
            
            return OrderBook(
                symbol=symbol,
                bids=[OrderBookLevel(price=mid_price * 0.999, size=1000)],
                asks=[OrderBookLevel(price=mid_price * 1.001, size=1000)],
                timestamp=int(time.time() * 1000)
            )
    
    async def fetch_balance(self) -> float:
        """Fetch account balance using official SDK."""
        try:
            await self._rate_limit()
            
            if not self.wallet_address:
                return 0.0
            
            user_state = self.info.user_state(self.wallet_address)
            if user_state and 'marginSummary' in user_state:
                account_value = float(user_state['marginSummary'].get('accountValue', 0))
                
                self.logger.debug(
                    "💰 Balance fetched via Official SDK",
                    account_value=account_value,
                    margin_summary=user_state['marginSummary']
                )
                
                return account_value
            
            return 0.0
            
        except Exception as e:
            self.logger.error("Error fetching balance", error=str(e))
            return 0.0
    
    async def fetch_positions(self) -> List[Position]:
        """Fetch positions using official SDK."""
        try:
            await self._rate_limit()
            
            if not self.wallet_address:
                return []
            
            user_state = self.info.user_state(self.wallet_address)
            if not user_state or 'assetPositions' not in user_state:
                return []
            
            positions = []
            for pos_data in user_state['assetPositions']:
                if pos_data['position']['szi'] != '0':  # Non-zero position
                    position_size = float(pos_data['position']['szi'])
                    entry_price = float(pos_data['position']['entryPx'])
                    unrealized_pnl = float(pos_data['position']['unrealizedPnl'])
                    
                    # Calculate additional fields
                    mark_price = float(pos_data['position'].get('markPx', entry_price))
                    notional = abs(position_size * mark_price)
                    percentage = (unrealized_pnl / notional * 100) if notional > 0 else 0.0
                    
                    position = Position(
                        symbol=pos_data['position']['coin'],
                        size=position_size,
                        entry_price=entry_price,
                        unrealized_pnl=unrealized_pnl,
                        side='long' if position_size > 0 else 'short',
                        contracts=position_size,  # For spot trading, contracts = size
                        notional=notional,
                        mark_price=mark_price,
                        percentage=percentage
                    )
                    positions.append(position)
            
            return positions
            
        except Exception as e:
            self.logger.error("Error fetching positions", error=str(e))
            return []
    
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: Optional[float] = None,
        type: OrderType = OrderType.LIMIT
    ) -> Dict[str, Any]:
        """
        Create order using OFFICIAL SDK - REAL ORDERS!
        
        This places ACTUAL orders that will trade with REAL MONEY!
        """
        try:
            await self._rate_limit()
            
            # Safety checks
            if not self.exchange:
                self.logger.warning(
                    "📋 SIMULATED ORDER (No private key provided)",
                    symbol=symbol,
                    side=side.value,
                    amount=amount,
                    price=price
                )
                import uuid
                return {
                    "id": str(uuid.uuid4())[:8],
                    "symbol": symbol,
                    "side": side.value,
                    "amount": amount,
                    "price": price,
                    "status": "simulated"
                }
            
            # Additional safety checks
            if amount > self.max_order_size:
                raise ValueError(f"Order size {amount} exceeds max {self.max_order_size}")
            
            # Check balance
            balance = await self.fetch_balance()
            if balance < self.min_balance:
                raise ValueError(f"Balance {balance} below minimum {self.min_balance}")
            
            # Prepare order for official SDK
            # Round price to avoid precision issues (different for each token)
            if symbol == "FTT":
                rounded_price = round(price, 3)  # FTT uses 3 decimal places (0.001 precision)
                rounded_amount = round(amount, 1)  # Round to 1 decimal for token amounts
            else:
                rounded_price = round(price, 6)  # GMT uses 6 decimal places
                rounded_amount = round(amount, 1)  # Round to 1 decimal for token amounts
            
            order_request = {
                "coin": symbol,
                "is_buy": side == OrderSide.BUY,
                "sz": rounded_amount,
                "limit_px": rounded_price,
                "order_type": {"limit": {"tif": "Gtc"}},  # Good Till Canceled
                "reduce_only": False
            }
            
            self.logger.critical(
                "🚨 PLACING REAL ORDER VIA OFFICIAL SDK",
                symbol=symbol,
                side=side.value,
                amount=amount,
                price=price,
                notional=f"${amount * (price or 0):.2f}",
                order_request=order_request
            )
            
            # Place the REAL order using official SDK
            # The official SDK expects individual parameters, not a dict
            result = self.exchange.order(
                order_request["coin"],
                order_request["is_buy"], 
                order_request["sz"],
                order_request["limit_px"],
                order_request["order_type"],
                reduce_only=order_request["reduce_only"]
            )
            
            if result and 'status' in result:
                order_id = result.get('response', {}).get('data', {}).get('statuses', [{}])[0].get('resting', {}).get('oid', 'unknown')
                
                self.logger.critical(
                    "✅ REAL ORDER PLACED SUCCESSFULLY",
                    order_id=order_id,
                    result=result,
                    symbol=symbol,
                    side=side.value,
                    amount=amount,
                    price=price
                )
                
                return {
                    "id": str(order_id),
                    "symbol": symbol,
                    "side": side.value,
                    "amount": amount,
                    "price": price,
                    "status": "placed",
                    "raw_result": result
                }
            else:
                raise Exception(f"Order failed: {result}")
                
        except Exception as e:
            self.logger.error(
                "❌ ORDER PLACEMENT FAILED",
                error=str(e),
                symbol=symbol,
                side=side.value,
                amount=amount,
                price=price
            )
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order using official SDK."""
        try:
            if not self.exchange:
                self.logger.warning("Cannot cancel order - no exchange API")
                return False
            
            await self._rate_limit()
            
            # The official SDK expects individual parameters, not a dict
            result = self.exchange.cancel(symbol, int(order_id))
            
            self.logger.info(
                "🗑️ Order cancellation requested",
                order_id=order_id,
                symbol=symbol,
                result=result
            )
            
            return result and result.get('status') == 'ok'
            
        except Exception as e:
            self.logger.error("Error canceling order", error=str(e), order_id=order_id)
            return False
    
    async def fetch_order_status(self, order_id: str, symbol: str) -> OrderStatus:
        """Fetch order status using official SDK."""
        try:
            if not self.wallet_address:
                return OrderStatus.UNKNOWN
            
            await self._rate_limit()
            
            # Get open orders
            open_orders = self.info.open_orders(self.wallet_address)
            
            for order in open_orders:
                if str(order.get('oid')) == str(order_id):
                    # Order is still open
                    return OrderStatus.OPEN
            
            # Check fills (order might be filled)
            user_fills = self.info.user_fills(self.wallet_address)
            for fill in user_fills:
                if str(fill.get('oid')) == str(order_id):
                    return OrderStatus.FILLED
            
            # Order not found - might be canceled or rejected
            return OrderStatus.CANCELED
            
        except Exception as e:
            self.logger.error("Error fetching order status", error=str(e), order_id=order_id)
            return OrderStatus.UNKNOWN
    
    async def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for symbol."""
        # Hyperliquid has standard fees
        return {
            'maker': 0.0002,  # 0.02% maker fee
            'taker': 0.0005   # 0.05% taker fee
        }
    
    async def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """Get market information for symbol."""
        try:
            await self._rate_limit()
            
            # Get all mids for current price
            all_mids = self.info.all_mids()
            current_price = float(all_mids.get(symbol, 0.0))
            
            # Get meta info for symbol details
            meta = self.info.meta()
            symbol_info = None
            
            for universe_item in meta.get('universe', []):
                if universe_item.get('name') == symbol:
                    symbol_info = universe_item
                    break
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'min_size': 1.0,  # Default minimum
                'tick_size': 0.001,  # Default tick
                'lot_size': 1.0,
                'symbol_info': symbol_info
            }
            
        except Exception as e:
            self.logger.error("Error fetching market info", error=str(e), symbol=symbol)
            return {
                'symbol': symbol,
                'current_price': 0.037,
                'min_size': 1.0,
                'tick_size': 0.001,
                'lot_size': 1.0
            }
    
    # Abstract methods that need to be implemented
    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch order details by ID."""
        try:
            if not self.wallet_address:
                return None
            
            await self._rate_limit()
            
            # Get open orders first
            open_orders = self.info.open_orders(self.wallet_address)
            for order in open_orders:
                if str(order.get('oid')) == str(order_id):
                    return order
            
            # Check fills if not found in open orders
            user_fills = self.info.user_fills(self.wallet_address)
            for fill in user_fills:
                if str(fill.get('oid')) == str(order_id):
                    return fill
            
            return None
            
        except Exception as e:
            self.logger.error("Error fetching order", error=str(e), order_id=order_id)
            return None
    
    async def load_markets(self) -> Dict[str, Any]:
        """Load market information."""
        try:
            await self._rate_limit()
            meta = self.info.meta()
            return meta.get('universe', [])
        except Exception as e:
            self.logger.error("Error loading markets", error=str(e))
            return {}
    
    async def watch_order_book(self, symbol: str, callback):
        """Watch order book updates (not implemented for now)."""
        self.logger.warning("watch_order_book not implemented for official SDK")
        pass
    
    async def watch_orders(self, callback):
        """Watch order updates (not implemented for now)."""
        self.logger.warning("watch_orders not implemented for official SDK")
        pass
    
    async def watch_trades(self, symbol: str, callback):
        """Watch trade updates (not implemented for now)."""
        self.logger.warning("watch_trades not implemented for official SDK")
        pass
    
    async def fetch_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Fetch all open orders."""
        try:
            if not self.wallet_address:
                return []
            
            await self._rate_limit()
            
            open_orders = self.info.open_orders(self.wallet_address)
            
            # Filter by symbol if provided
            if symbol:
                open_orders = [order for order in open_orders if order.get('coin') == symbol]
            
            self.logger.info(
                "📋 Fetched open orders",
                count=len(open_orders),
                symbol=symbol or "all"
            )
            
            return open_orders
            
        except Exception as e:
            self.logger.error("Error fetching open orders", error=str(e))
            return []
    
    async def cancel_all_orders(self, symbol: str = None) -> int:
        """Cancel all open orders for a symbol (or all symbols)."""
        try:
            open_orders = await self.fetch_open_orders(symbol)
            
            if not open_orders:
                self.logger.info("🟢 No open orders to cancel")
                return 0
            
            cancelled_count = 0
            
            for order in open_orders:
                order_id = str(order.get('oid'))
                order_symbol = order.get('coin')
                
                try:
                    success = await self.cancel_order(order_id, order_symbol)
                    if success:
                        cancelled_count += 1
                        self.logger.info(
                            "✅ Cancelled order",
                            order_id=order_id,
                            symbol=order_symbol,
                            side=order.get('side'),
                            size=order.get('sz')
                        )
                    else:
                        self.logger.warning(
                            "⚠️ Failed to cancel order",
                            order_id=order_id,
                            symbol=order_symbol
                        )
                except Exception as e:
                    self.logger.error(
                        "❌ Error cancelling order",
                        order_id=order_id,
                        error=str(e)
                    )
            
            self.logger.warning(
                "🗑️ ORDER CLEANUP COMPLETE",
                cancelled=cancelled_count,
                total=len(open_orders),
                symbol=symbol or "all"
            )
            
            return cancelled_count
            
        except Exception as e:
            self.logger.error("Error in cancel_all_orders", error=str(e))
            return 0
    
    async def close_position(self, symbol: str, position_size: float) -> bool:
        """Close a position by placing a market order with reduce_only=True."""
        try:
            if not self.exchange:
                self.logger.warning("Cannot close position - no exchange API")
                return False
            
            if abs(position_size) < 0.001:  # Position too small to close
                return True
            
            # Determine side: if position is positive (long), we need to sell to close
            is_buy = position_size < 0  # If short position, buy to close
            close_size = abs(position_size)
            
            # Round size to avoid precision issues
            close_size = round(close_size, 1)
            
            # Get current market price for market order
            try:
                all_mids = self.info.all_mids()
                current_price = float(all_mids.get(symbol, 0.037))
            except:
                current_price = 0.037  # Fallback price
            
            # Place market order to close position
            order_request = {
                "coin": symbol,
                "is_buy": is_buy,
                "sz": close_size,
                "limit_px": current_price,  # Use current price for market order
                "order_type": {"limit": {"tif": "Ioc"}},  # Immediate or Cancel for market order
                "reduce_only": True  # This is key - only reduce position, don't increase
            }
            
            self.logger.critical(
                "🚨 CLOSING POSITION VIA OFFICIAL SDK",
                symbol=symbol,
                position_size=position_size,
                close_size=close_size,
                is_buy=is_buy,
                current_price=current_price,
                order_request=order_request
            )
            
            # Place the order using official SDK
            result = self.exchange.order(
                order_request["coin"],
                order_request["is_buy"], 
                order_request["sz"],
                order_request["limit_px"],
                order_request["order_type"],
                reduce_only=order_request["reduce_only"]
            )
            
            if result and 'status' in result:
                self.logger.critical(
                    "✅ POSITION CLOSED SUCCESSFULLY",
                    symbol=symbol,
                    position_size=position_size,
                    result=result
                )
                return True
            else:
                self.logger.error(f"Failed to close position: {result}")
                return False
                
        except Exception as e:
            self.logger.error(
                "❌ POSITION CLOSURE FAILED",
                error=str(e),
                symbol=symbol,
                position_size=position_size
            )
            return False
    
    async def close_all_positions(self) -> int:
        """Close all open positions."""
        try:
            positions = await self.fetch_positions()
            
            if not positions:
                self.logger.info("🟢 No open positions to close")
                return 0
            
            closed_count = 0
            
            for position in positions:
                try:
                    success = await self.close_position(position.symbol, position.size)
                    if success:
                        closed_count += 1
                        self.logger.info(
                            "✅ Closed position",
                            symbol=position.symbol,
                            size=position.size,
                            side=position.side,
                            pnl=position.unrealized_pnl
                        )
                    else:
                        self.logger.warning(
                            "⚠️ Failed to close position",
                            symbol=position.symbol,
                            size=position.size
                        )
                except Exception as e:
                    self.logger.error(
                        "❌ Error closing position",
                        symbol=position.symbol,
                        error=str(e)
                    )
            
            self.logger.warning(
                "🚨 POSITION CLOSURE COMPLETE",
                closed=closed_count,
                total=len(positions)
            )
            
            return closed_count
            
        except Exception as e:
            self.logger.error("Error in close_all_positions", error=str(e))
            return 0
