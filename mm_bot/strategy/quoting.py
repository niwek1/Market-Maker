"""
Quote generation and price adjustment utilities.

This module provides utilities for generating and adjusting quotes based on
various market conditions and constraints.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from mm_bot.logging import LoggerMixin
from mm_bot.utils.safemath import round_to_tick, round_to_lot, clamp
from mm_bot.strategy.avellaneda_stoikov import ASQuote


@dataclass
class QuoteConstraints:
    """Constraints for quote generation."""
    min_spread_bps: float = 5.0
    max_spread_bps: float = 500.0
    tick_size: float = 0.01
    lot_size: float = 0.001
    min_notional: float = 10.0
    max_notional: Optional[float] = None


@dataclass
class AdjustedQuote:
    """Quote adjusted for market constraints."""
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    spread_bps: float
    valid: bool = True
    rejection_reason: Optional[str] = None


class QuoteAdjuster(LoggerMixin):
    """Adjust quotes based on market constraints and exchange requirements."""
    
    def __init__(self):
        super().__init__()
    
    def adjust_quote(
        self,
        quote: ASQuote,
        constraints: QuoteConstraints,
        mid_price: float,
        available_balance: float = float('inf')
    ) -> AdjustedQuote:
        """
        Adjust quote to meet exchange and risk constraints.
        
        Args:
            quote: Original Avellaneda-Stoikov quote
            constraints: Quote constraints
            mid_price: Current mid price
            available_balance: Available balance for quoting
            
        Returns:
            Adjusted quote
        """
        try:
            # Start with original quote
            bid_price = quote.bid_price
            ask_price = quote.ask_price
            bid_size = quote.bid_size
            ask_size = quote.ask_size
            
            # Adjust prices to tick size
            bid_price = round_to_tick(bid_price, constraints.tick_size)
            ask_price = round_to_tick(ask_price, constraints.tick_size)
            
            # Ensure minimum spread
            current_spread_bps = ((ask_price - bid_price) / mid_price) * 10000
            if current_spread_bps < constraints.min_spread_bps:
                # Widen spread symmetrically
                required_spread = (constraints.min_spread_bps / 10000) * mid_price
                current_spread = ask_price - bid_price
                spread_adjustment = (required_spread - current_spread) / 2
                
                bid_price = round_to_tick(bid_price - spread_adjustment, constraints.tick_size)
                ask_price = round_to_tick(ask_price + spread_adjustment, constraints.tick_size)
                current_spread_bps = ((ask_price - bid_price) / mid_price) * 10000
            
            # Ensure maximum spread
            if current_spread_bps > constraints.max_spread_bps:
                # Narrow spread symmetrically
                required_spread = (constraints.max_spread_bps / 10000) * mid_price
                current_spread = ask_price - bid_price
                spread_adjustment = (current_spread - required_spread) / 2
                
                bid_price = round_to_tick(bid_price + spread_adjustment, constraints.tick_size)
                ask_price = round_to_tick(ask_price - spread_adjustment, constraints.tick_size)
                current_spread_bps = ((ask_price - bid_price) / mid_price) * 10000
            
            # Adjust sizes to lot size
            bid_size = round_to_lot(bid_size, constraints.lot_size)
            ask_size = round_to_lot(ask_size, constraints.lot_size)
            
            # Check minimum notional
            bid_notional = bid_price * bid_size
            ask_notional = ask_price * ask_size
            
            if bid_notional < constraints.min_notional:
                bid_size = round_to_lot(
                    constraints.min_notional / bid_price,
                    constraints.lot_size
                )
                bid_notional = bid_price * bid_size
            
            if ask_notional < constraints.min_notional:
                ask_size = round_to_lot(
                    constraints.min_notional / ask_price,
                    constraints.lot_size
                )
                ask_notional = ask_price * ask_size
            
            # Check maximum notional
            if constraints.max_notional:
                if bid_notional > constraints.max_notional:
                    bid_size = round_to_lot(
                        constraints.max_notional / bid_price,
                        constraints.lot_size
                    )
                    bid_notional = bid_price * bid_size
                
                if ask_notional > constraints.max_notional:
                    ask_size = round_to_lot(
                        constraints.max_notional / ask_price,
                        constraints.lot_size
                    )
                    ask_notional = ask_price * ask_size
            
            # Check available balance
            total_required = bid_notional + ask_notional
            if total_required > available_balance:
                # Scale down sizes proportionally
                scale_factor = available_balance / total_required
                bid_size = round_to_lot(bid_size * scale_factor, constraints.lot_size)
                ask_size = round_to_lot(ask_size * scale_factor, constraints.lot_size)
                
                # Recalculate notionals
                bid_notional = bid_price * bid_size
                ask_notional = ask_price * ask_size
            
            # AGGRESSIVE validation - allow more orders through
            valid = True
            rejection_reason = None
            
            # Only reject if completely invalid
            if bid_size <= 0 or ask_size <= 0:
                valid = False
                rejection_reason = "Zero size"
            elif bid_price >= ask_price:
                valid = False  
                rejection_reason = "Crossed quote"
            # REMOVED: min_notional check to be more aggressive
            # Allow small orders through for active trading
            
            return AdjustedQuote(
                bid_price=bid_price,
                ask_price=ask_price,
                bid_size=bid_size,
                ask_size=ask_size,
                spread_bps=current_spread_bps,
                valid=valid,
                rejection_reason=rejection_reason
            )
            
        except Exception as e:
            self.logger.error("Error adjusting quote", error=str(e))
            return AdjustedQuote(
                bid_price=quote.bid_price,
                ask_price=quote.ask_price,
                bid_size=0.0,
                ask_size=0.0,
                spread_bps=0.0,
                valid=False,
                rejection_reason=f"Adjustment error: {e}"
            )
    
    def apply_fee_adjustment(
        self,
        quote: AdjustedQuote,
        maker_fee_bps: float,
        target_edge_bps: float = 5.0
    ) -> AdjustedQuote:
        """
        Adjust quote prices to account for trading fees.
        
        Args:
            quote: Quote to adjust
            maker_fee_bps: Maker fee in basis points
            target_edge_bps: Target edge after fees
            
        Returns:
            Fee-adjusted quote
        """
        if not quote.valid:
            return quote
        
        try:
            # Calculate required price adjustment
            total_edge_bps = maker_fee_bps * 2 + target_edge_bps  # Both sides pay fees
            
            mid_price = (quote.bid_price + quote.ask_price) / 2
            adjustment = (total_edge_bps / 10000) * mid_price / 2
            
            # Adjust prices
            adjusted_bid = quote.bid_price - adjustment
            adjusted_ask = quote.ask_price + adjustment
            
            # Recalculate spread
            adjusted_spread_bps = ((adjusted_ask - adjusted_bid) / mid_price) * 10000
            
            return AdjustedQuote(
                bid_price=adjusted_bid,
                ask_price=adjusted_ask,
                bid_size=quote.bid_size,
                ask_size=quote.ask_size,
                spread_bps=adjusted_spread_bps,
                valid=quote.valid,
                rejection_reason=quote.rejection_reason
            )
            
        except Exception as e:
            self.logger.error("Error applying fee adjustment", error=str(e))
            return quote


def calculate_inventory_adjusted_sizes(
    base_size: float,
    inventory: float,
    max_inventory: float,
    inventory_target: float = 0.0,
    max_adjustment: float = 0.5
) -> Tuple[float, float]:
    """
    Calculate inventory-adjusted quote sizes.
    
    Args:
        base_size: Base quote size
        inventory: Current inventory
        max_inventory: Maximum inventory
        inventory_target: Target inventory level
        max_adjustment: Maximum size adjustment factor
        
    Returns:
        Tuple of (bid_size, ask_size)
    """
    if max_inventory <= 0:
        return base_size, base_size
    
    # Calculate inventory ratio
    inventory_ratio = (inventory - inventory_target) / max_inventory
    inventory_ratio = clamp(inventory_ratio, -1.0, 1.0)
    
    # Adjust sizes based on inventory
    # When long (positive inventory), reduce bid size and increase ask size
    # When short (negative inventory), increase bid size and reduce ask size
    size_adjustment = inventory_ratio * max_adjustment
    
    bid_size = base_size * (1.0 - size_adjustment)
    ask_size = base_size * (1.0 + size_adjustment)
    
    # Ensure sizes are positive
    bid_size = max(0.0, bid_size)
    ask_size = max(0.0, ask_size)
    
    return bid_size, ask_size


def calculate_volatility_adjusted_spread(
    base_spread: float,
    current_volatility: float,
    reference_volatility: float = 0.02,  # 2% reference vol
    vol_adjustment_factor: float = 0.5
) -> float:
    """
    Adjust spread based on volatility regime.
    
    Args:
        base_spread: Base spread
        current_volatility: Current volatility estimate
        reference_volatility: Reference volatility level
        vol_adjustment_factor: Volatility adjustment strength
        
    Returns:
        Volatility-adjusted spread
    """
    if reference_volatility <= 0:
        return base_spread
    
    # Calculate volatility ratio
    vol_ratio = current_volatility / reference_volatility
    
    # Apply adjustment
    adjustment = (vol_ratio - 1.0) * vol_adjustment_factor
    adjusted_spread = base_spread * (1.0 + adjustment)
    
    return max(0.0, adjusted_spread)
