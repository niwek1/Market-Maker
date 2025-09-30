use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    pub max_position: Decimal,
    pub max_notional: Decimal,
    pub max_drawdown_pct: f64,
    pub max_order_size: Decimal,
    pub max_daily_volume: Decimal,
    pub max_order_rate_per_min: u32,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_position: Decimal::from(1000),    // $1000 max position
            max_notional: Decimal::from(50),      // $50 per order
            max_drawdown_pct: 5.0,                // 5% max drawdown
            max_order_size: Decimal::from(100),   // $100 max order size
            max_daily_volume: Decimal::from(10000), // $10k daily volume limit
            max_order_rate_per_min: 60,           // 60 orders per minute
        }
    }
}

#[derive(Debug, Clone)]
pub struct RiskManager {
    limits: RiskLimits,
    current_position: Decimal,
    daily_volume: Decimal,
    order_timestamps: Vec<DateTime<Utc>>,
    starting_balance: Decimal,
    peak_balance: Decimal,
}

impl RiskManager {
    pub fn new(limits: RiskLimits, starting_balance: Decimal) -> Self {
        Self {
            limits,
            current_position: Decimal::ZERO,
            daily_volume: Decimal::ZERO,
            order_timestamps: Vec::new(),
            starting_balance,
            peak_balance: starting_balance,
        }
    }

    pub fn check_order_risk(
        &mut self,
        side: OrderSide,
        size: Decimal,
        price: Decimal,
        current_balance: Decimal,
    ) -> Result<()> {
        let notional = size * price;
        
        // Check order size limits
        if size > self.limits.max_order_size {
            anyhow::bail!("Order size {} exceeds maximum {}", size, self.limits.max_order_size);
        }

        if notional > self.limits.max_notional {
            anyhow::bail!("Order notional {} exceeds maximum {}", notional, self.limits.max_notional);
        }

        // Check position limits
        let new_position = match side {
            OrderSide::Buy => self.current_position + size,
            OrderSide::Sell => self.current_position - size,
        };

        if new_position.abs() > self.limits.max_position {
            anyhow::bail!(
                "New position {} would exceed maximum {}",
                new_position.abs(),
                self.limits.max_position
            );
        }

        // Check drawdown limits
        self.update_peak_balance(current_balance);
        let drawdown_pct = self.calculate_drawdown_pct(current_balance);
        if drawdown_pct > self.limits.max_drawdown_pct {
            anyhow::bail!(
                "Drawdown {:.2}% exceeds maximum {:.2}%",
                drawdown_pct,
                self.limits.max_drawdown_pct
            );
        }

        // Check daily volume limits
        if self.daily_volume + notional > self.limits.max_daily_volume {
            anyhow::bail!(
                "Daily volume {} would exceed maximum {}",
                self.daily_volume + notional,
                self.limits.max_daily_volume
            );
        }

        // Check order rate limits
        self.cleanup_old_timestamps();
        if self.order_timestamps.len() >= self.limits.max_order_rate_per_min as usize {
            anyhow::bail!(
                "Order rate {} orders/min exceeds maximum {}",
                self.order_timestamps.len(),
                self.limits.max_order_rate_per_min
            );
        }

        Ok(())
    }

    pub fn record_order(&mut self, size: Decimal, price: Decimal) {
        let notional = size * price;
        self.daily_volume += notional;
        self.order_timestamps.push(Utc::now());
    }

    pub fn update_position(&mut self, new_position: Decimal) {
        self.current_position = new_position;
    }

    pub fn get_position(&self) -> Decimal {
        self.current_position
    }

    pub fn get_daily_volume(&self) -> Decimal {
        self.daily_volume
    }

    pub fn calculate_drawdown_pct(&self, current_balance: Decimal) -> f64 {
        if self.peak_balance <= Decimal::ZERO {
            return 0.0;
        }

        let drawdown = self.peak_balance - current_balance;
        (drawdown / self.peak_balance).to_f64().unwrap_or(0.0) * 100.0
    }

    fn update_peak_balance(&mut self, current_balance: Decimal) {
        if current_balance > self.peak_balance {
            self.peak_balance = current_balance;
        }
    }

    fn cleanup_old_timestamps(&mut self) {
        let cutoff = Utc::now() - chrono::Duration::minutes(1);
        self.order_timestamps.retain(|&ts| ts > cutoff);
    }

    pub fn reset_daily_volume(&mut self) {
        self.daily_volume = Decimal::ZERO;
    }

    pub fn is_risk_ok(&self, current_balance: Decimal) -> bool {
        let drawdown_pct = self.calculate_drawdown_pct(current_balance);
        drawdown_pct <= self.limits.max_drawdown_pct
    }

    pub fn get_risk_summary(&self, current_balance: Decimal) -> RiskSummary {
        RiskSummary {
            current_position: self.current_position,
            daily_volume: self.daily_volume,
            drawdown_pct: self.calculate_drawdown_pct(current_balance),
            orders_last_minute: self.order_timestamps.len() as u32,
            is_risk_ok: self.is_risk_ok(current_balance),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RiskSummary {
    pub current_position: Decimal,
    pub daily_volume: Decimal,
    pub drawdown_pct: f64,
    pub orders_last_minute: u32,
    pub is_risk_ok: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}
