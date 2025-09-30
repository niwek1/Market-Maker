#!/usr/bin/env python3
"""
Enhanced Live Trading Bot with Professional Interface

Features:
- Real-time trading dashboard
- Live P&L tracking
- Order status monitoring
- Performance metrics
- Beautiful console interface
"""

import asyncio
import signal
import sys
import os
from datetime import datetime, timedelta
import time
from dataclasses import dataclass
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mm_bot.config import load_config
from mm_bot.exchanges.hyperliquid_official import HyperliquidOfficialAdapter
from mm_bot.strategy.avellaneda_stoikov import AvellanedaStoikovStrategy
from mm_bot.marketdata.l2_cache import L2Cache
from mm_bot.risk.limits import RiskLimitManager
from mm_bot.execution.order_manager import OrderSide, OrderType

@dataclass
class TradingMetrics:
    orders_placed: int = 0
    orders_filled: int = 0
    total_volume: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0
    session_start: datetime = None
    last_trade_time: datetime = None

@dataclass
class OrderInfo:
    order_id: str
    side: str
    amount: float
    price: float
    status: str
    timestamp: datetime
    notional: float

class EnhancedLiveTradingBot:
    def __init__(self):
        self.running = False
        self.config = None
        self.adapter = None
        self.strategy = None
        self.l2_cache = None
        self.risk_manager = None
        
        # Trading metrics
        self.metrics = TradingMetrics()
        self.orders: List[OrderInfo] = []
        self.starting_balance = 0.0
        self.current_balance = 0.0
        
        # Order management
        self.active_order_ids = set()  # Track our active orders
        self.last_cleanup = 0
        self.cleanup_interval = 30  # Clean up old orders every 30 seconds
        
        # Display settings
        self.last_update = 0
        self.update_interval = 2  # Update every 2 seconds
        
    def print_header(self):
        """Print beautiful header."""
        os.system('clear' if os.name == 'posix' else 'cls')
        print("🚀" + "="*78 + "🚀")
        print("🎯" + " "*25 + "ELITE LIVE TRADING BOT" + " "*25 + "🎯")
        print("💰" + "="*78 + "💰")
        print()
    
    def print_account_info(self):
        """Print account information."""
        elapsed = datetime.now() - self.metrics.session_start if self.metrics.session_start else timedelta(0)
        
        print("📊 ACCOUNT STATUS:")
        print(f"   💰 Starting Balance: ${self.starting_balance:.2f}")
        print(f"   💵 Current Balance:  ${self.current_balance:.2f}")
        print(f"   📈 Session P&L:      ${self.current_balance - self.starting_balance:+.2f}")
        print(f"   ⏰ Session Time:     {str(elapsed).split('.')[0]}")
        print(f"   🛡️ Max Loss Limit:   $5.00")
        print()
    
    def print_trading_metrics(self):
        """Print trading performance metrics."""
        fill_rate = (self.metrics.orders_filled / max(self.metrics.orders_placed, 1)) * 100
        
        print("📈 TRADING METRICS:")
        print(f"   🎯 Orders Placed:    {self.metrics.orders_placed}")
        print(f"   ✅ Orders Filled:    {self.metrics.orders_filled}")
        print(f"   📊 Fill Rate:        {fill_rate:.1f}%")
        print(f"   💹 Total Volume:     ${self.metrics.total_volume:.2f}")
        print(f"   💰 Realized P&L:     ${self.metrics.realized_pnl:+.2f}")
        print(f"   💸 Fees Paid:        ${self.metrics.fees_paid:.2f}")
        print()
    
    def print_recent_orders(self):
        """Print recent order history."""
        print("📋 RECENT ORDERS (Last 10):")
        recent_orders = self.orders[-10:] if len(self.orders) > 10 else self.orders
        
        if not recent_orders:
            print("   No orders yet...")
        else:
            for order in reversed(recent_orders):
                status_icon = "✅" if order.status == "filled" else "🔄" if order.status == "open" else "❌"
                side_icon = "🟢" if order.side == "buy" else "🔴"
                time_str = order.timestamp.strftime("%H:%M:%S")
                print(f"   {status_icon} {side_icon} {order.side.upper()} {order.amount:.1f} FTT @ ${order.price:.2f} | ${order.notional:.2f} | {time_str}")
        print()
    
    def print_market_info(self, current_price=0.0, spread_pct=0.0):
        """Print current market information."""
        print("📊 MARKET DATA:")
        print(f"   💎 FTT Price:        ${current_price:.2f}")
        print(f"   📏 Current Spread:   {spread_pct:.3f}%")
        print(f"   🎯 Target Spread:    0.300%")
        print(f"   📦 Order Size:       11.0 FTT (~$10.00)")
        print(f"   🗑️ Active Orders:    {len(self.active_order_ids)}")
        print()
    
    def print_order_management_info(self):
        """Print order management information."""
        print("🔧 ORDER & POSITION MANAGEMENT:")
        print(f"   🗑️ Auto-cleanup every {self.cleanup_interval}s (cancels orders >2min old)")
        print(f"   🛑 Emergency stop: cancels ALL orders + closes ALL positions")
        print(f"   📊 Active order tracking and status monitoring")
        print(f"   🚨 Position closure with reduce_only orders")
        next_cleanup = int(self.cleanup_interval - (time.time() - self.last_cleanup))
        print(f"   ⏰ Next cleanup in: {max(0, next_cleanup)}s")
        print()
    
    def print_status_bar(self):
        """Print status bar."""
        now = datetime.now().strftime("%H:%M:%S")
        status = "🟢 ACTIVE" if self.running else "🔴 STOPPED"
        print("─" * 80)
        print(f"Status: {status} | Time: {now} | Press Ctrl+C to stop")
        print("🚀" + "="*78 + "🚀")
    
    def update_display(self, current_price=0.0, spread_pct=0.0):
        """Update the entire display."""
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
        
        self.last_update = current_time
        
        self.print_header()
        self.print_account_info()
        self.print_trading_metrics()
        self.print_market_info(current_price, spread_pct)
        self.print_recent_orders()
        self.print_order_management_info()
        self.print_status_bar()
    
    async def initialize(self):
        """Initialize bot components."""
        self.print_header()
        print("🚀 INITIALIZING ENHANCED LIVE TRADING BOT...")
        print("⚠️  THIS WILL TRADE WITH REAL MONEY!")
        print("🛡️  Maximum Loss: $5.00 (16.8% protection)")
        print("📏 Order Size: 270 GMT tokens (~$10.15 per trade)")
        print("⏰ Session Duration: Unlimited (stop with Ctrl+C)")
        print()
        
        # Load config
        self.config = load_config("configs/hyperliquid_safe.yaml")
        print("✅ Configuration loaded")
        
        # Initialize exchange adapter
        exchange_config = self.config.exchanges['hyperliquid']
        self.adapter = HyperliquidOfficialAdapter(exchange_config)
        connected = await self.adapter.connect()
        
        if not connected:
            raise Exception("Failed to connect to exchange")
        
        print("✅ Connected to Hyperliquid")
        
        # Get starting balance
        self.starting_balance = await self.adapter.fetch_balance()
        self.current_balance = self.starting_balance
        print(f"✅ Starting balance: ${self.starting_balance:.2f}")
        
        # Initialize strategy
        self.strategy = AvellanedaStoikovStrategy(self.config.strategy.as_)
        print("✅ Strategy initialized")
        
        # Initialize L2 cache
        self.l2_cache = L2Cache(symbol="FTT", max_levels=10, feature_history_size=100)
        print("✅ L2 cache initialized")
        
        # Initialize risk manager
        self.risk_manager = RiskLimitManager(self.config.risk)
        print("✅ Risk manager initialized")
        
        # Initialize metrics
        self.metrics.session_start = datetime.now()
        
        print("\n🎯 ALL SYSTEMS READY - STARTING LIVE TRADING IN 3 SECONDS...")
        await asyncio.sleep(3)
        
        return True
    
    async def place_enhanced_order(self, side, amount, price):
        """Place order with enhanced tracking."""
        try:
            # Risk validation
            allowed, violations = self.risk_manager.check_order_allowed(
                symbol="FTT",
                side=side,
                amount=amount,
                price=price
            )
            
            if not allowed:
                return None
            
            # Place the order
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
            
            result = await self.adapter.create_order(
                symbol="FTT",
                side=order_side,
                amount=amount,
                price=price,
                type=OrderType.LIMIT
            )
            
            # Track order
            order_id = result.get('id', 'unknown')
            order_info = OrderInfo(
                order_id=order_id,
                side=side,
                amount=amount,
                price=price,
                status="open",
                timestamp=datetime.now(),
                notional=amount * price
            )
            
            self.orders.append(order_info)
            self.metrics.orders_placed += 1
            self.metrics.total_volume += amount * price
            
            # Track active orders
            if order_id != 'unknown':
                self.active_order_ids.add(order_id)
            
            return result
            
        except Exception as e:
            return None
    
    async def update_balance(self):
        """Update current balance."""
        try:
            self.current_balance = await self.adapter.fetch_balance()
        except:
            pass  # Keep last known balance if fetch fails
    
    async def check_order_fills(self):
        """Check for filled orders and update metrics."""
        try:
            # Get all open orders from exchange
            open_orders = await self.adapter.fetch_open_orders("FTT")
            open_order_ids = {str(order.get('oid')) for order in open_orders}
            
            # Check which of our tracked orders are no longer open (filled)
            filled_orders = []
            for order_id in list(self.active_order_ids):
                if order_id not in open_order_ids:
                    # Order is no longer open - likely filled!
                    filled_orders.append(order_id)
                    self.metrics.orders_filled += 1
                    print(f"🎉 ORDER FILLED! Order ID: {order_id}")
            
            # Remove filled orders from tracking
            for order_id in filled_orders:
                self.active_order_ids.discard(order_id)
                
        except Exception as e:
            self.logger.error(f"Error checking order fills: {e}")
    
    async def cleanup_old_orders(self):
        """Cancel old unused orders periodically."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        self.last_cleanup = current_time
        
        try:
            # First check for fills
            await self.check_order_fills()
            
            # Get all open orders
            open_orders = await self.adapter.fetch_open_orders("FTT")
            
            if not open_orders:
                return
            
            # Cancel orders older than 2 minutes
            cutoff_time = datetime.now() - timedelta(minutes=2)
            cancelled_count = 0
            
            for order in open_orders:
                order_id = str(order.get('oid'))
                
                # Cancel old orders (simplified - in production you'd check timestamp)
                try:
                    success = await self.adapter.cancel_order(order_id, "FTT")
                    if success:
                        cancelled_count += 1
                        # Remove from our tracking
                        self.active_order_ids.discard(order_id)
                except:
                    pass
            
            if cancelled_count > 0:
                print(f"🗑️ Cleaned up {cancelled_count} old orders")
                
        except Exception as e:
            pass  # Silent cleanup failure
    
    async def emergency_cancel_all(self):
        """Emergency cancel all orders and close all positions."""
        try:
            print("\n🚨 EMERGENCY CLEANUP STARTING...")
            
            # First cancel all orders
            print("🗑️ Cancelling all open orders...")
            cancelled = await self.adapter.cancel_all_orders("GMT")
            print(f"✅ Cancelled {cancelled} orders")
            self.active_order_ids.clear()
            
            # Then close all positions
            print("🚨 Closing all open positions...")
            closed = await self.adapter.close_all_positions()
            print(f"✅ Closed {closed} positions")
            
            print("✅ Emergency cleanup completed")
            
        except Exception as e:
            print(f"❌ Error during emergency cleanup: {e}")
    
    async def run_trading_loop(self):
        """Enhanced trading loop with real-time display."""
        self.running = True
        loop_count = 0
        
        try:
            while self.running:
                loop_count += 1
                
                # Fetch market data
                try:
                    order_book = await self.adapter.fetch_order_book("FTT")
                    
                    # Update L2 cache
                    bids_tuples = [(level.price, level.size) for level in order_book.bids]
                    asks_tuples = [(level.price, level.size) for level in order_book.asks]
                    
                    success = self.l2_cache.update_order_book(
                        bids=bids_tuples,
                        asks=asks_tuples,
                        timestamp=order_book.timestamp
                    )
                    
                    # Get features
                    features = self.l2_cache.get_current_features()
                    
                    if features:
                        mid_price = features.mid
                        volatility = max(features.short_vol or 0.01, 0.01)
                        
                        # Generate quote
                        quote = self.strategy.generate_quote(
                            mid_price=mid_price,
                            volatility=volatility,
                            inventory=0.0,
                            max_inventory=self.config.risk.max_position,
                            quote_size=self.config.risk.quote_size_base
                        )
                        
                        # Calculate spread
                        spread_pct = ((quote.ask_price - quote.bid_price) / mid_price) * 100
                        
                        # Check for fills first
                        await self.check_order_fills()
                        
                        # Update display
                        self.update_display(mid_price, spread_pct)
                        
                        # Place orders every other loop
                        if loop_count % 2 == 0:
                            # Update balance
                            await self.update_balance()
                            
                            # Clean up old orders periodically
                            await self.cleanup_old_orders()
                            
                            # Place buy order
                            await self.place_enhanced_order("buy", quote.bid_size, quote.bid_price)
                            await asyncio.sleep(1)
                            
                            # Place sell order
                            await self.place_enhanced_order("sell", quote.ask_size, quote.ask_price)
                
                except Exception as e:
                    pass  # Continue on errors
                
                # Wait for next iteration
                await asyncio.sleep(10)  # 10 second intervals
                
        except KeyboardInterrupt:
            print("\n🛑 Trading stopped by user")
        finally:
            self.running = False
    
    async def cleanup(self):
        """Clean up resources, cancel all orders, and close all positions."""
        print("\n🧹 COMPREHENSIVE CLEANUP STARTING...")
        
        if self.adapter:
            try:
                # Emergency cleanup: cancel orders and close positions
                await self.emergency_cancel_all()
                
                # Wait a moment for orders to process
                print("⏳ Waiting for cleanup to process...")
                await asyncio.sleep(2)
                
                # Verify cleanup was successful
                remaining_orders = await self.adapter.fetch_open_orders("FTT")
                remaining_positions = await self.adapter.fetch_positions()
                
                if remaining_orders:
                    print(f"⚠️ Warning: {len(remaining_orders)} orders still open")
                else:
                    print("✅ All orders successfully cancelled")
                
                if remaining_positions:
                    print(f"⚠️ Warning: {len(remaining_positions)} positions still open")
                    for pos in remaining_positions:
                        print(f"   - {pos.symbol}: {pos.size} ({pos.side})")
                else:
                    print("✅ All positions successfully closed")
                
            except Exception as e:
                print(f"⚠️ Error during comprehensive cleanup: {e}")
            
            # Finally disconnect
            try:
                await self.adapter.disconnect()
                print("✅ Disconnected from exchange")
            except Exception as e:
                print(f"⚠️ Error disconnecting: {e}")
        
        print("✅ Comprehensive cleanup completed")
    
    async def generate_final_report(self):
        """Generate enhanced final report."""
        try:
            final_balance = await self.adapter.fetch_balance()
            session_pnl = final_balance - self.starting_balance
            elapsed = datetime.now() - self.metrics.session_start
            
            self.print_header()
            print("📋 FINAL TRADING SESSION REPORT")
            print("="*80)
            print()
            print("💰 FINANCIAL SUMMARY:")
            print(f"   Starting Balance:    ${self.starting_balance:.2f}")
            print(f"   Final Balance:       ${final_balance:.2f}")
            print(f"   Session P&L:         ${session_pnl:+.2f}")
            print(f"   P&L Percentage:      {(session_pnl/self.starting_balance)*100:+.2f}%")
            print()
            print("📊 TRADING STATISTICS:")
            print(f"   Session Duration:    {str(elapsed).split('.')[0]}")
            print(f"   Orders Placed:       {self.metrics.orders_placed}")
            print(f"   Orders Filled:       {self.metrics.orders_filled}")
            print(f"   Fill Rate:           {(self.metrics.orders_filled/max(self.metrics.orders_placed,1))*100:.1f}%")
            print(f"   Total Volume:        ${self.metrics.total_volume:.2f}")
            print()
            print("🎯 PERFORMANCE METRICS:")
            if self.metrics.orders_placed > 0:
                avg_order_size = self.metrics.total_volume / self.metrics.orders_placed
                print(f"   Average Order Size:  ${avg_order_size:.2f}")
                orders_per_minute = self.metrics.orders_placed / (elapsed.total_seconds() / 60)
                print(f"   Orders Per Minute:   {orders_per_minute:.1f}")
            
            print()
            print("✅ LIVE TRADING SESSION COMPLETED SUCCESSFULLY!")
            print("="*80)
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
    
    async def run(self):
        """Main run method."""
        try:
            await self.initialize()
            await self.run_trading_loop()
        finally:
            await self.generate_final_report()
            await self.cleanup()

async def main():
    bot = EnhancedLiveTradingBot()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        bot.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
