"""
Avellaneda-Stoikov market making strategy implementation.

This module implements the classic Avellaneda-Stoikov optimal market making
framework with inventory-aware pricing and optimal spread calculation.

References:
- Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book.
  Quantitative Finance, 8(3), 217-224.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar

from mm_bot.config import AvellanedaStoikovConfig
from mm_bot.logging import LoggerMixin
from mm_bot.utils.safemath import safe_divide, safe_log, safe_sqrt, clamp


@dataclass
class ASQuote:
    """Avellaneda-Stoikov quote with metadata."""
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    reservation_price: float
    optimal_spread: float
    inventory_skew: float
    timestamp: int
    
    @property
    def mid_price(self) -> float:
        """Get mid price of the quote."""
        return (self.bid_price + self.ask_price) / 2.0
    
    @property
    def spread(self) -> float:
        """Get spread of the quote."""
        return self.ask_price - self.bid_price
    
    @property
    def spread_bps(self) -> float:
        """Get spread in basis points."""
        mid = self.mid_price
        return (self.spread / mid) * 10000 if mid > 0 else 0


@dataclass
class ASParams:
    """Avellaneda-Stoikov parameters for a given state."""
    gamma: float  # Risk aversion
    kappa: float  # Inventory penalty
    sigma: float  # Volatility
    T: float      # Time to horizon
    inventory: float  # Current inventory
    inventory_target: float  # Target inventory
    
    def __post_init__(self):
        """Validate parameters."""
        if self.gamma <= 0:
            raise ValueError("Risk aversion (gamma) must be positive")
        if self.kappa < 0:
            raise ValueError("Inventory penalty (kappa) must be non-negative")
        if self.sigma <= 0:
            raise ValueError("Volatility (sigma) must be positive")
        if self.T <= 0:
            raise ValueError("Time horizon (T) must be positive")


class AvellanedaStoikovStrategy(LoggerMixin):
    """
    Avellaneda-Stoikov optimal market making strategy.
    
    This implementation provides:
    - Optimal bid/ask pricing based on inventory and market conditions
    - Dynamic spread calculation
    - Inventory mean reversion
    - Risk-adjusted pricing
    """
    
    def __init__(self, config: AvellanedaStoikovConfig):
        """
        Initialize Avellaneda-Stoikov strategy.
        
        Args:
            config: Strategy configuration
        """
        super().__init__()
        self.config = config
        
        # Strategy state
        self.current_inventory = 0.0
        self.last_mid_price = 0.0
        self.volatility_estimate = 0.01  # Default 1% volatility
        
        # Performance tracking
        self.quotes_generated = 0
        self.inventory_history = []
        
        self.logger.info(
            "Initialized Avellaneda-Stoikov strategy",
            gamma=config.gamma,
            kappa=config.kappa,
            inventory_target=config.inventory_target,
            inventory_band=config.inventory_band
        )
    
    def calculate_reservation_price(
        self,
        mid_price: float,
        inventory: float,
        volatility: float,
        time_horizon: float = 1.0
    ) -> float:
        """
        Calculate reservation price (indifference price).
        
        The reservation price is the price at which the market maker is
        indifferent between holding and not holding an additional unit.
        
        r = S - (γσ²T/2) * (q - q̄)
        
        Args:
            mid_price: Current mid price
            inventory: Current inventory position
            volatility: Volatility estimate
            time_horizon: Time horizon (fraction of day, typically small)
            
        Returns:
            Reservation price
        """
        inventory_deviation = inventory - self.config.inventory_target
        
        # Risk adjustment term
        risk_adjustment = (
            self.config.gamma * 
            (volatility ** 2) * 
            time_horizon * 
            inventory_deviation / 2.0
        )
        
        reservation_price = mid_price - risk_adjustment
        
        self.logger.debug(
            "Calculated reservation price",
            mid_price=mid_price,
            inventory=inventory,
            inventory_deviation=inventory_deviation,
            risk_adjustment=risk_adjustment,
            reservation_price=reservation_price
        )
        
        return reservation_price
    
    def calculate_optimal_spread(
        self,
        volatility: float,
        time_horizon: float = 1.0,
        arrival_rate: float = 1.0
    ) -> float:
        """
        Calculate optimal half-spread.
        
        The optimal spread balances the trade-off between adverse selection
        and inventory risk.
        
        δ* = γσ²T + (2/γ) * ln(1 + γ/κ)
        
        Args:
            volatility: Volatility estimate
            time_horizon: Time horizon
            arrival_rate: Order arrival rate (trades per unit time)
            
        Returns:
            Optimal half-spread
        """
        gamma = self.config.gamma
        kappa = max(self.config.kappa, 1e-6)  # Avoid division by zero
        
        # Inventory risk component
        inventory_component = gamma * (volatility ** 2) * time_horizon
        
        # Adverse selection component
        # Using approximation for computational stability
        if gamma / kappa < 0.1:  # Small argument approximation
            adverse_selection_component = (2.0 / gamma) * (gamma / kappa)
        else:
            adverse_selection_component = (2.0 / gamma) * safe_log(1 + gamma / kappa, default=0)
        
        optimal_half_spread = inventory_component + adverse_selection_component
        
        # Apply minimum spread constraint
        min_half_spread = volatility * 0.5  # At least 0.5 * volatility
        optimal_half_spread = max(optimal_half_spread, min_half_spread)
        
        # OVERRIDE: Force reasonable spreads for low-priced tokens (GMT fix)
        # If the calculated spread is insane, override with a percentage-based spread
        max_reasonable_half_spread = 0.01  # 1% half spread = 2% full spread
        if optimal_half_spread > max_reasonable_half_spread:
            # Use percentage-based spread instead of A-S formula
            target_spread_pct = getattr(self.config, 'target_spread_pct', 0.002)  # Default 0.2%
            # NOTE: This will be converted to absolute in generate_quote() using mid_price
            optimal_half_spread = target_spread_pct / 2  # Store as percentage for now
            
            self.logger.warning(
                "🔧 SPREAD OVERRIDE: A-S formula gave unrealistic spread, using percentage-based",
                original_half_spread=inventory_component + adverse_selection_component,
                override_half_spread=optimal_half_spread,
                target_spread_pct=target_spread_pct * 100
            )
        
        self.logger.debug(
            "Calculated optimal spread",
            volatility=volatility,
            gamma=gamma,
            kappa=kappa,
            inventory_component=inventory_component,
            adverse_selection_component=adverse_selection_component,
            optimal_half_spread=optimal_half_spread
        )
        
        return optimal_half_spread
    
    def calculate_inventory_skew(
        self,
        inventory: float,
        max_inventory: float
    ) -> float:
        """
        Calculate inventory-based skew for asymmetric quoting.
        
        When inventory is high, we want to quote more aggressively on the sell side
        and less aggressively on the buy side, and vice versa.
        
        Args:
            inventory: Current inventory
            max_inventory: Maximum allowed inventory
            
        Returns:
            Skew factor (-1 to 1, negative = favor selling, positive = favor buying)
        """
        if max_inventory <= 0:
            return 0.0
        
        # Normalize inventory to [-1, 1] range
        inventory_ratio = clamp(
            inventory / max_inventory,
            -1.0,
            1.0
        )
        
        # Apply inventory band - only skew when outside target band
        band_size = self.config.inventory_band
        target = self.config.inventory_target / max_inventory if max_inventory > 0 else 0
        
        # Calculate distance from target band
        if inventory_ratio > target + band_size:
            # Too long, favor selling
            skew = -(inventory_ratio - target - band_size) / (1.0 - target - band_size)
        elif inventory_ratio < target - band_size:
            # Too short, favor buying  
            skew = (target - band_size - inventory_ratio) / (target - band_size + 1.0)
        else:
            # Within band, no skew
            skew = 0.0
        
        return clamp(skew, -1.0, 1.0)
    
    def generate_quote(
        self,
        mid_price: float,
        volatility: float,
        inventory: float,
        max_inventory: float,
        quote_size: float,
        time_horizon: float = 1.0 / (24 * 60),  # 1 minute in days
        timestamp: Optional[int] = None
    ) -> ASQuote:
        """
        Generate Avellaneda-Stoikov optimal quote.
        
        Args:
            mid_price: Current mid price
            volatility: Volatility estimate
            inventory: Current inventory
            max_inventory: Maximum inventory limit
            quote_size: Size to quote on each side
            time_horizon: Time horizon for optimization
            timestamp: Quote timestamp
            
        Returns:
            ASQuote with optimal bid/ask prices
        """
        if timestamp is None:
            import time
            timestamp = int(time.time() * 1000)
        
        # Apply volatility scaling and floor from config for safety
        vol_floor = getattr(self.config, 'vol_floor', 0.01)
        vol_scaling = getattr(self.config, 'vol_scaling', 1.0)
        
        # Ensure minimum volatility and apply scaling for conservative spreads
        volatility = max(volatility, vol_floor) * vol_scaling
        
        # Update internal state
        self.current_inventory = inventory
        self.last_mid_price = mid_price
        self.volatility_estimate = volatility
        
        # Calculate reservation price
        reservation_price = self.calculate_reservation_price(
            mid_price, inventory, volatility, time_horizon
        )
        
        # Calculate optimal spread
        optimal_half_spread = self.calculate_optimal_spread(
            volatility, time_horizon
        )
        
        # Convert percentage-based spread to absolute spread using mid_price
        # If optimal_half_spread is very small (< 0.01), it's likely a percentage that needs conversion
        if optimal_half_spread <= 0.01:
            # Convert percentage to absolute using mid_price
            # optimal_half_spread is already the half-spread percentage (e.g., 0.001 = 0.1%)
            optimal_half_spread = mid_price * optimal_half_spread  # Convert percentage to absolute
            
            self.logger.debug(
                "🔧 CONVERTED percentage spread to absolute",
                mid_price=mid_price,
                percentage_half_spread=optimal_half_spread / mid_price,
                absolute_half_spread=optimal_half_spread
            )
        
        # Calculate inventory skew
        inventory_skew = self.calculate_inventory_skew(inventory, max_inventory)
        
        # Apply inventory skew to spread
        # When skew is negative (too long), we tighten ask and widen bid
        # When skew is positive (too short), we tighten bid and widen ask
        skew_adjustment = optimal_half_spread * inventory_skew * 0.5
        
        bid_spread = optimal_half_spread + skew_adjustment
        ask_spread = optimal_half_spread - skew_adjustment
        
        # Ensure minimum spreads (but not for percentage-based overrides)
        # If we're using percentage-based spreads, don't apply volatility-based minimums
        if optimal_half_spread > 0.01:  # Original A-S calculation
            min_spread = volatility * 0.1
            bid_spread = max(bid_spread, min_spread)
            ask_spread = max(ask_spread, min_spread)
        else:  # Percentage-based override - respect the calculated spreads
            # Allow very tight spreads for active trading
            min_spread = optimal_half_spread * 0.1  # 10% of calculated spread as absolute minimum
            bid_spread = max(bid_spread, min_spread)
            ask_spread = max(ask_spread, min_spread)
        
        # Calculate final prices
        bid_price = reservation_price - bid_spread
        ask_price = reservation_price + ask_spread
        
        # Create quote
        quote = ASQuote(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=quote_size,
            ask_size=quote_size,
            reservation_price=reservation_price,
            optimal_spread=optimal_half_spread * 2,  # Full spread
            inventory_skew=inventory_skew,
            timestamp=timestamp
        )
        
        self.quotes_generated += 1
        self.inventory_history.append(inventory)
        
        self.logger.debug(
            "Generated AS quote",
            mid_price=mid_price,
            reservation_price=reservation_price,
            bid_price=bid_price,
            ask_price=ask_price,
            optimal_spread=quote.optimal_spread,
            inventory_skew=inventory_skew,
            inventory=inventory
        )
        
        return quote
    
    def update_inventory(self, new_inventory: float) -> None:
        """Update current inventory position."""
        self.current_inventory = new_inventory
        self.inventory_history.append(new_inventory)
    
    def get_inventory_target_adjustment(
        self,
        current_pnl: float,
        volatility_regime: str = "normal"
    ) -> float:
        """
        Calculate dynamic inventory target based on performance and market regime.
        
        Args:
            current_pnl: Current PnL
            volatility_regime: Market volatility regime
            
        Returns:
            Adjusted inventory target
        """
        base_target = self.config.inventory_target
        
        # Adjust based on PnL (reduce target if losing money)
        if current_pnl < 0:
            pnl_adjustment = min(0.5, abs(current_pnl) / 1000.0)  # Scale by notional
            base_target *= (1.0 - pnl_adjustment)
        
        # Adjust based on volatility regime
        vol_adjustment = {
            "low": 1.2,      # Can hold more inventory in low vol
            "normal": 1.0,
            "high": 0.7,     # Reduce inventory in high vol
            "extreme": 0.5   # Minimize inventory in extreme vol
        }.get(volatility_regime, 1.0)
        
        adjusted_target = base_target * vol_adjustment
        
        self.logger.debug(
            "Calculated inventory target adjustment",
            base_target=base_target,
            pnl_adjustment=current_pnl,
            vol_regime=volatility_regime,
            adjusted_target=adjusted_target
        )
        
        return adjusted_target
    
    def calculate_fair_value_adjustment(
        self,
        order_book_imbalance: float,
        trade_flow_imbalance: float,
        microprice: float,
        mid_price: float
    ) -> float:
        """
        Calculate fair value adjustment based on microstructure signals.
        
        Args:
            order_book_imbalance: Order book volume imbalance (-1 to 1)
            trade_flow_imbalance: Recent trade flow imbalance (-1 to 1)  
            microprice: Volume-weighted microprice
            mid_price: Simple mid price
            
        Returns:
            Fair value adjustment (added to reservation price)
        """
        # Weight different signals
        ob_weight = 0.3
        flow_weight = 0.4
        microprice_weight = 0.3
        
        # Order book signal
        ob_signal = order_book_imbalance * ob_weight
        
        # Trade flow signal  
        flow_signal = trade_flow_imbalance * flow_weight
        
        # Microprice signal
        microprice_signal = 0.0
        if mid_price > 0:
            price_diff = (microprice - mid_price) / mid_price
            microprice_signal = clamp(price_diff * 10, -0.5, 0.5) * microprice_weight
        
        # Combine signals
        total_adjustment = ob_signal + flow_signal + microprice_signal
        
        # Scale by volatility (larger adjustments in high vol markets)
        vol_scaling = min(2.0, self.volatility_estimate / 0.01)  # Scale relative to 1% vol
        adjustment = total_adjustment * self.volatility_estimate * vol_scaling
        
        self.logger.debug(
            "Calculated fair value adjustment",
            ob_imbalance=order_book_imbalance,
            flow_imbalance=trade_flow_imbalance,
            microprice_diff=microprice - mid_price,
            adjustment=adjustment
        )
        
        return adjustment
    
    def get_strategy_stats(self) -> dict:
        """Get strategy performance statistics."""
        stats = {
            "quotes_generated": self.quotes_generated,
            "current_inventory": self.current_inventory,
            "last_mid_price": self.last_mid_price,
            "volatility_estimate": self.volatility_estimate,
            "config": {
                "gamma": self.config.gamma,
                "kappa": self.config.kappa,
                "inventory_target": self.config.inventory_target,
                "inventory_band": self.config.inventory_band,
            }
        }
        
        if self.inventory_history:
            inventory_array = np.array(self.inventory_history[-100:])  # Last 100 observations
            stats["inventory_stats"] = {
                "mean": float(np.mean(inventory_array)),
                "std": float(np.std(inventory_array)),
                "min": float(np.min(inventory_array)),
                "max": float(np.max(inventory_array)),
            }
        
        return stats
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.current_inventory = 0.0
        self.last_mid_price = 0.0
        self.volatility_estimate = 0.01
        self.quotes_generated = 0
        self.inventory_history.clear()
        
        self.logger.info("Reset Avellaneda-Stoikov strategy state")
