"""
Safe mathematical operations for financial calculations.
"""

import math
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_EVEN
from typing import Union, Optional

import numpy as np


# Type alias for numeric types
Numeric = Union[int, float, Decimal]


def safe_divide(
    numerator: Numeric,
    denominator: Numeric,
    default: Optional[Numeric] = None
) -> Optional[Numeric]:
    """
    Safe division that handles zero denominator.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def safe_log(x: Numeric, default: Optional[Numeric] = None) -> Optional[Numeric]:
    """
    Safe logarithm that handles non-positive values.
    
    Args:
        x: Input value
        default: Default value for non-positive inputs
        
    Returns:
        Natural logarithm or default value
    """
    if x <= 0:
        return default
    return math.log(x)


def safe_sqrt(x: Numeric, default: Optional[Numeric] = None) -> Optional[Numeric]:
    """
    Safe square root that handles negative values.
    
    Args:
        x: Input value
        default: Default value for negative inputs
        
    Returns:
        Square root or default value
    """
    if x < 0:
        return default
    return math.sqrt(x)


def round_to_tick(price: float, tick_size: float) -> float:
    """
    Round price to nearest tick size.
    
    Args:
        price: Price to round
        tick_size: Minimum price increment
        
    Returns:
        Rounded price
    """
    if tick_size <= 0:
        return price
    
    return round(price / tick_size) * tick_size


def round_to_lot(size: float, lot_size: float) -> float:
    """
    Round size to nearest lot size.
    
    Args:
        size: Size to round
        lot_size: Minimum size increment
        
    Returns:
        Rounded size
    """
    if lot_size <= 0:
        return size
    
    return round(size / lot_size) * lot_size


def floor_to_tick(price: float, tick_size: float) -> float:
    """
    Floor price to tick size boundary.
    
    Args:
        price: Price to floor
        tick_size: Minimum price increment
        
    Returns:
        Floored price
    """
    if tick_size <= 0:
        return price
    
    return math.floor(price / tick_size) * tick_size


def ceil_to_tick(price: float, tick_size: float) -> float:
    """
    Ceil price to tick size boundary.
    
    Args:
        price: Price to ceil
        tick_size: Minimum price increment
        
    Returns:
        Ceiled price
    """
    if tick_size <= 0:
        return price
    
    return math.ceil(price / tick_size) * tick_size


def floor_to_lot(size: float, lot_size: float) -> float:
    """
    Floor size to lot size boundary.
    
    Args:
        size: Size to floor
        lot_size: Minimum size increment
        
    Returns:
        Floored size
    """
    if lot_size <= 0:
        return size
    
    return math.floor(size / lot_size) * lot_size


def ceil_to_lot(size: float, lot_size: float) -> float:
    """
    Ceil size to lot size boundary.
    
    Args:
        size: Size to ceil
        lot_size: Minimum size increment
        
    Returns:
        Ceiled size
    """
    if lot_size <= 0:
        return size
    
    return math.ceil(size / lot_size) * lot_size


def precision_from_tick_size(tick_size: float) -> int:
    """
    Calculate decimal precision from tick size.
    
    Args:
        tick_size: Minimum price increment
        
    Returns:
        Number of decimal places
    """
    if tick_size <= 0:
        return 0
    
    # Convert to string and count decimal places
    tick_str = f"{tick_size:.10f}".rstrip('0').rstrip('.')
    if '.' in tick_str:
        return len(tick_str.split('.')[1])
    return 0


def format_price(price: float, tick_size: float) -> str:
    """
    Format price with appropriate precision based on tick size.
    
    Args:
        price: Price to format
        tick_size: Minimum price increment
        
    Returns:
        Formatted price string
    """
    precision = precision_from_tick_size(tick_size)
    return f"{price:.{precision}f}"


def format_size(size: float, lot_size: float) -> str:
    """
    Format size with appropriate precision based on lot size.
    
    Args:
        size: Size to format
        lot_size: Minimum size increment
        
    Returns:
        Formatted size string
    """
    precision = precision_from_tick_size(lot_size)
    return f"{size:.{precision}f}"


def clamp(value: Numeric, min_value: Numeric, max_value: Numeric) -> Numeric:
    """
    Clamp value between min and max bounds.
    
    Args:
        value: Value to clamp
        min_value: Minimum bound
        max_value: Maximum bound
        
    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))


def is_close(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    """
    Check if two floating point numbers are close.
    
    Args:
        a: First number
        b: Second number
        rel_tol: Relative tolerance
        abs_tol: Absolute tolerance
        
    Returns:
        True if numbers are close
    """
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def basis_points_to_decimal(bps: float) -> float:
    """
    Convert basis points to decimal.
    
    Args:
        bps: Basis points
        
    Returns:
        Decimal equivalent
    """
    return bps / 10000.0


def decimal_to_basis_points(decimal: float) -> float:
    """
    Convert decimal to basis points.
    
    Args:
        decimal: Decimal value
        
    Returns:
        Basis points equivalent
    """
    return decimal * 10000.0


def percentage_to_decimal(pct: float) -> float:
    """
    Convert percentage to decimal.
    
    Args:
        pct: Percentage
        
    Returns:
        Decimal equivalent
    """
    return pct / 100.0


def decimal_to_percentage(decimal: float) -> float:
    """
    Convert decimal to percentage.
    
    Args:
        decimal: Decimal value
        
    Returns:
        Percentage equivalent
    """
    return decimal * 100.0


def compound_return(returns: list[float]) -> float:
    """
    Calculate compound return from a series of returns.
    
    Args:
        returns: List of period returns
        
    Returns:
        Compound return
    """
    if not returns:
        return 0.0
    
    compound = 1.0
    for r in returns:
        compound *= (1.0 + r)
    
    return compound - 1.0


def annualized_return(total_return: float, periods: int, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return.
    
    Args:
        total_return: Total return over the period
        periods: Number of periods
        periods_per_year: Periods per year (default: 252 for daily)
        
    Returns:
        Annualized return
    """
    if periods <= 0:
        return 0.0
    
    years = periods / periods_per_year
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: List of period returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Periods per year
        
    Returns:
        Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - (risk_free_rate / periods_per_year)
    
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0:
        return 0.0
    
    return mean_excess / std_excess * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation).
    
    Args:
        returns: List of period returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Periods per year
        
    Returns:
        Sortino ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - (risk_free_rate / periods_per_year)
    
    mean_excess = np.mean(excess_returns)
    
    # Calculate downside deviation
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0:
        return float('inf') if mean_excess > 0 else 0.0
    
    downside_deviation = np.sqrt(np.mean(negative_returns ** 2))
    
    if downside_deviation == 0:
        return 0.0
    
    return mean_excess / downside_deviation * np.sqrt(periods_per_year)


def max_drawdown(values: list[float]) -> tuple[float, int, int]:
    """
    Calculate maximum drawdown.
    
    Args:
        values: List of portfolio values
        
    Returns:
        Tuple of (max_drawdown, start_index, end_index)
    """
    if not values:
        return 0.0, 0, 0
    
    values_array = np.array(values)
    peak = np.maximum.accumulate(values_array)
    drawdown = (values_array - peak) / peak
    
    max_dd_idx = np.argmin(drawdown)
    max_dd = drawdown[max_dd_idx]
    
    # Find the peak before the max drawdown
    peak_idx = np.argmax(values_array[:max_dd_idx + 1])
    
    return abs(max_dd), peak_idx, max_dd_idx


def value_at_risk(returns: list[float], confidence: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: List of returns
        confidence: Confidence level (e.g., 0.05 for 95% VaR)
        
    Returns:
        VaR value (positive number representing loss)
    """
    if not returns:
        return 0.0
    
    returns_array = np.array(returns)
    var = -np.percentile(returns_array, confidence * 100)
    
    return max(0.0, var)  # Return positive value for loss


def expected_shortfall(returns: list[float], confidence: float = 0.05) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR).
    
    Args:
        returns: List of returns
        confidence: Confidence level
        
    Returns:
        Expected shortfall value
    """
    if not returns:
        return 0.0
    
    returns_array = np.array(returns)
    var_threshold = -np.percentile(returns_array, confidence * 100)
    
    tail_returns = returns_array[returns_array <= -var_threshold]
    
    if len(tail_returns) == 0:
        return 0.0
    
    return -np.mean(tail_returns)
