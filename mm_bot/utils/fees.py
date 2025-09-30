"""
Fee calculation utilities for different exchanges and order types.
"""

from typing import Dict, Optional, Tuple
from enum import Enum

from mm_bot.utils.safemath import basis_points_to_decimal


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class FeeCalculator:
    """Fee calculation for different exchanges."""
    
    # Default fee schedules (in basis points)
    DEFAULT_FEES = {
        "binance": {
            "spot": {"maker": 10, "taker": 10},
            "futures": {"maker": 2, "taker": 4},
        },
        "kraken": {
            "spot": {"maker": 16, "taker": 26},
            "futures": {"maker": 2, "taker": 5},
        },
        "coinbase": {
            "spot": {"maker": 50, "taker": 50},
            "futures": {"maker": 40, "taker": 60},
        },
        "ftx": {
            "spot": {"maker": 2, "taker": 7},
            "futures": {"maker": 2, "taker": 7},
        },
        "okx": {
            "spot": {"maker": 8, "taker": 10},
            "futures": {"maker": 2, "taker": 5},
        },
    }
    
    def __init__(
        self,
        exchange: str,
        market_type: str = "spot",
        custom_fees: Optional[Dict[str, float]] = None
    ):
        """
        Initialize fee calculator.
        
        Args:
            exchange: Exchange name
            market_type: Market type (spot, futures, etc.)
            custom_fees: Custom fee schedule override
        """
        self.exchange = exchange.lower()
        self.market_type = market_type.lower()
        
        if custom_fees:
            self.fees = custom_fees
        else:
            self.fees = self.DEFAULT_FEES.get(self.exchange, {}).get(
                self.market_type, {"maker": 10, "taker": 10}
            )
    
    def calculate_fee(
        self,
        size: float,
        price: float,
        order_type: OrderType,
        is_maker: bool = True
    ) -> float:
        """
        Calculate trading fee.
        
        Args:
            size: Order size
            price: Order price
            order_type: Order type
            is_maker: Whether the order provides liquidity (maker)
            
        Returns:
            Fee amount in quote currency
        """
        notional = size * price
        
        if is_maker and order_type == OrderType.LIMIT:
            fee_rate = basis_points_to_decimal(self.fees["maker"])
        else:
            fee_rate = basis_points_to_decimal(self.fees["taker"])
        
        return notional * fee_rate
    
    def calculate_fee_bps(
        self,
        order_type: OrderType,
        is_maker: bool = True
    ) -> float:
        """
        Get fee rate in basis points.
        
        Args:
            order_type: Order type
            is_maker: Whether the order provides liquidity
            
        Returns:
            Fee rate in basis points
        """
        if is_maker and order_type == OrderType.LIMIT:
            return self.fees["maker"]
        else:
            return self.fees["taker"]
    
    def net_proceeds(
        self,
        size: float,
        price: float,
        side: OrderSide,
        order_type: OrderType,
        is_maker: bool = True
    ) -> float:
        """
        Calculate net proceeds after fees.
        
        Args:
            size: Order size
            price: Order price
            side: Order side (buy/sell)
            order_type: Order type
            is_maker: Whether the order provides liquidity
            
        Returns:
            Net proceeds (positive for sells, negative for buys)
        """
        notional = size * price
        fee = self.calculate_fee(size, price, order_type, is_maker)
        
        if side == OrderSide.SELL:
            return notional - fee
        else:
            return -(notional + fee)
    
    def breakeven_spread(
        self,
        price: float,
        order_type: OrderType = OrderType.LIMIT
    ) -> float:
        """
        Calculate minimum spread to break even on fees.
        
        Args:
            price: Current price
            order_type: Order type
            
        Returns:
            Minimum spread in price units
        """
        maker_fee_bps = self.calculate_fee_bps(order_type, is_maker=True)
        taker_fee_bps = self.calculate_fee_bps(order_type, is_maker=False)
        
        # Need to cover both maker fees (buy and sell)
        total_fee_bps = 2 * maker_fee_bps
        
        return price * basis_points_to_decimal(total_fee_bps)
    
    def effective_spread(
        self,
        bid: float,
        ask: float,
        order_type: OrderType = OrderType.LIMIT
    ) -> float:
        """
        Calculate effective spread after accounting for fees.
        
        Args:
            bid: Bid price
            ask: Ask price
            order_type: Order type
            
        Returns:
            Effective spread in price units
        """
        mid = (bid + ask) / 2
        nominal_spread = ask - bid
        fee_cost = self.breakeven_spread(mid, order_type)
        
        return max(0, nominal_spread - fee_cost)


class FundingCalculator:
    """Funding rate calculation for perpetual futures."""
    
    def __init__(self, exchange: str):
        self.exchange = exchange.lower()
    
    def calculate_funding_cost(
        self,
        position_size: float,
        mark_price: float,
        funding_rate: float,
        hours: float = 8.0
    ) -> float:
        """
        Calculate funding cost for a position.
        
        Args:
            position_size: Position size (positive for long, negative for short)
            mark_price: Mark price
            funding_rate: Funding rate (8-hour rate)
            hours: Time period in hours
            
        Returns:
            Funding cost (negative = payment, positive = receipt)
        """
        notional = abs(position_size) * mark_price
        funding_periods = hours / 8.0  # Funding typically every 8 hours
        
        # Long positions pay funding when rate is positive
        # Short positions receive funding when rate is positive
        if position_size > 0:  # Long position
            return -notional * funding_rate * funding_periods
        elif position_size < 0:  # Short position
            return notional * funding_rate * funding_periods
        else:
            return 0.0
    
    def annualized_funding_rate(self, eight_hour_rate: float) -> float:
        """
        Convert 8-hour funding rate to annualized rate.
        
        Args:
            eight_hour_rate: 8-hour funding rate
            
        Returns:
            Annualized funding rate
        """
        # 3 funding periods per day, 365 days per year
        return eight_hour_rate * 3 * 365


def calculate_slippage_cost(
    size: float,
    market_price: float,
    executed_price: float,
    side: OrderSide
) -> float:
    """
    Calculate slippage cost.
    
    Args:
        size: Order size
        market_price: Market price at order time
        executed_price: Actual execution price
        side: Order side
        
    Returns:
        Slippage cost (always positive)
    """
    notional = size * market_price
    
    if side == OrderSide.BUY:
        # For buys, slippage is paying more than market price
        slippage_rate = max(0, (executed_price - market_price) / market_price)
    else:
        # For sells, slippage is receiving less than market price
        slippage_rate = max(0, (market_price - executed_price) / market_price)
    
    return notional * slippage_rate


def total_transaction_cost(
    size: float,
    price: float,
    side: OrderSide,
    order_type: OrderType,
    fee_calculator: FeeCalculator,
    executed_price: Optional[float] = None,
    is_maker: bool = True
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate total transaction cost including fees and slippage.
    
    Args:
        size: Order size
        price: Order price
        side: Order side
        order_type: Order type
        fee_calculator: Fee calculator instance
        executed_price: Actual execution price (for slippage)
        is_maker: Whether the order provides liquidity
        
    Returns:
        Tuple of (total_cost, cost_breakdown)
    """
    # Calculate fee
    fee_cost = fee_calculator.calculate_fee(size, price, order_type, is_maker)
    
    # Calculate slippage if execution price provided
    slippage_cost = 0.0
    if executed_price is not None:
        slippage_cost = calculate_slippage_cost(size, price, executed_price, side)
    
    total_cost = fee_cost + slippage_cost
    
    cost_breakdown = {
        "fee": fee_cost,
        "slippage": slippage_cost,
        "total": total_cost,
        "fee_bps": fee_calculator.calculate_fee_bps(order_type, is_maker),
        "slippage_bps": (slippage_cost / (size * price)) * 10000 if size * price > 0 else 0,
        "total_bps": (total_cost / (size * price)) * 10000 if size * price > 0 else 0,
    }
    
    return total_cost, cost_breakdown


def optimize_order_size(
    available_balance: float,
    price: float,
    fee_calculator: FeeCalculator,
    min_notional: float = 10.0,
    max_position_pct: float = 0.95
) -> float:
    """
    Optimize order size considering fees and balance constraints.
    
    Args:
        available_balance: Available balance
        price: Order price
        fee_calculator: Fee calculator
        min_notional: Minimum notional value
        max_position_pct: Maximum percentage of balance to use
        
    Returns:
        Optimal order size
    """
    # Calculate maximum affordable size considering fees
    max_balance_to_use = available_balance * max_position_pct
    
    # For buy orders, need to account for fees
    maker_fee_rate = basis_points_to_decimal(
        fee_calculator.calculate_fee_bps(OrderType.LIMIT, is_maker=True)
    )
    
    # Size * price * (1 + fee_rate) <= max_balance_to_use
    max_size = max_balance_to_use / (price * (1 + maker_fee_rate))
    
    # Check minimum notional constraint
    min_size = min_notional / price
    
    return max(min_size, max_size) if max_size >= min_size else 0.0
