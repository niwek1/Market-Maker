use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use anyhow::Result;

use crate::order_book::OrderBook;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvellanedaStoikovParams {
    pub gamma: f64,           // Risk aversion parameter
    pub kappa: f64,           // Adverse selection parameter
    pub inventory_target: f64, // Target inventory level
    pub inventory_band: f64,   // Inventory tolerance band
    pub vol_window_secs: u64,  // Volatility calculation window
    pub vol_floor: f64,        // Minimum volatility floor
}

impl Default for AvellanedaStoikovParams {
    fn default() -> Self {
        Self {
            gamma: 0.001,      // Very low risk aversion for tight spreads
            kappa: 0.05,       // Moderate adverse selection protection
            inventory_target: 0.0, // Stay neutral
            inventory_band: 0.1,   // 10% inventory tolerance
            vol_window_secs: 60,   // 1-minute volatility window
            vol_floor: 0.001,      // 0.1% minimum volatility
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptimalQuotes {
    pub bid_price: Decimal,
    pub ask_price: Decimal,
    pub bid_size: Decimal,
    pub ask_size: Decimal,
    pub reservation_price: Decimal,
    pub spread: Decimal,
    pub timestamp: DateTime<Utc>,
}

pub struct AvellanedaStoikovStrategy {
    params: AvellanedaStoikovParams,
    price_history: Vec<(DateTime<Utc>, Decimal)>,
    current_inventory: f64,
}

impl AvellanedaStoikovStrategy {
    pub fn new(params: AvellanedaStoikovParams) -> Self {
        Self {
            params,
            price_history: Vec::new(),
            current_inventory: 0.0,
        }
    }

    pub fn calculate_optimal_quotes(
        &mut self,
        order_book: &OrderBook,
        config: &crate::config::TradingConfig,
    ) -> Result<Option<OptimalQuotes>> {
        let mid_price = order_book.mid_price()
            .ok_or_else(|| anyhow::anyhow!("No mid price available"))?;

        // Update price history
        self.update_price_history(mid_price);

        // Calculate volatility
        let volatility = self.calculate_volatility();

        // Calculate reservation price
        let reservation_price = self.calculate_reservation_price(mid_price, volatility);

        // Calculate optimal spread
        let (bid_price, ask_price) = self.calculate_optimal_spread(
            mid_price,
            reservation_price,
            volatility,
            config,
        )?;

        // Calculate order sizes
        let (bid_size, ask_size) = self.calculate_order_sizes(
            mid_price,
            reservation_price,
            volatility,
            config,
        );

        let spread = ask_price - bid_price;

        Ok(Some(OptimalQuotes {
            bid_price,
            ask_price,
            bid_size,
            ask_size,
            reservation_price,
            spread,
            timestamp: Utc::now(),
        }))
    }

    fn update_price_history(&mut self, price: Decimal) {
        let now = Utc::now();
        self.price_history.push((now, price));

        // Keep only recent prices within the volatility window
        let cutoff_time = now - chrono::Duration::seconds(self.params.vol_window_secs as i64);
        self.price_history.retain(|(timestamp, _)| *timestamp > cutoff_time);
    }

    fn calculate_volatility(&self) -> f64 {
        if self.price_history.len() < 2 {
            return self.params.vol_floor;
        }

        let returns: Vec<f64> = self.price_history
            .windows(2)
            .map(|window| {
                let (_, price1) = window[0];
                let (_, price2) = window[1];
                (price2.to_f64().unwrap_or(0.0) / price1.to_f64().unwrap_or(1.0)).ln()
            })
            .collect();

        if returns.is_empty() {
            return self.params.vol_floor;
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;

        let volatility = variance.sqrt();
        volatility.max(self.params.vol_floor)
    }

    fn calculate_reservation_price(&self, mid_price: Decimal, volatility: f64) -> Decimal {
        let mid_f64 = mid_price.to_f64().unwrap_or(0.0);
        let inventory_deviation = self.current_inventory - self.params.inventory_target;
        
        // Risk adjustment based on inventory
        let risk_adjustment = self.params.gamma * volatility * inventory_deviation;
        
        let reservation_price = mid_f64 - risk_adjustment;
        Decimal::from_f64_retain(reservation_price).unwrap_or(mid_price)
    }

    fn calculate_optimal_spread(
        &self,
        mid_price: Decimal,
        reservation_price: Decimal,
        volatility: f64,
        config: &crate::config::TradingConfig,
    ) -> Result<(Decimal, Decimal)> {
        let mid_f64 = mid_price.to_f64().unwrap_or(0.0);
        let reservation_f64 = reservation_price.to_f64().unwrap_or(0.0);
        
        // Avellaneda-Stoikov optimal half-spread
        let adverse_selection_component = self.params.kappa * volatility;
        let inventory_component = self.params.gamma * volatility * 
            (self.current_inventory - self.params.inventory_target).abs();
        
        let optimal_half_spread = adverse_selection_component + inventory_component;
        
        // Apply spread override if configured
        let target_half_spread = if config.target_spread_pct > 0.0 {
            config.target_spread_pct / 2.0 / 100.0
        } else {
            optimal_half_spread
        };

        // Convert to absolute prices
        let half_spread_abs = mid_f64 * target_half_spread;
        
        let bid_price = reservation_f64 - half_spread_abs;
        let ask_price = reservation_f64 + half_spread_abs;

        Ok((
            Decimal::from_f64_retain(bid_price).unwrap_or(mid_price),
            Decimal::from_f64_retain(ask_price).unwrap_or(mid_price),
        ))
    }

    fn calculate_order_sizes(
        &self,
        _mid_price: Decimal,
        _reservation_price: Decimal,
        _volatility: f64,
        config: &crate::config::TradingConfig,
    ) -> (Decimal, Decimal) {
        let base_size = config.order_size;
        
        // Adjust size based on inventory
        let inventory_skew = (self.current_inventory - self.params.inventory_target) / 
            self.params.inventory_band;
        
        let skew_factor = (-inventory_skew).exp().min(2.0).max(0.5);
        
        let adjusted_size = base_size * Decimal::from_f64_retain(skew_factor).unwrap_or(Decimal::ONE);
        
        (adjusted_size, adjusted_size)
    }

    pub fn update_inventory(&mut self, inventory: f64) {
        self.current_inventory = inventory;
    }

    pub fn get_inventory(&self) -> f64 {
        self.current_inventory
    }
}
