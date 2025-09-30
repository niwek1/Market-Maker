use serde::{Deserialize, Serialize};
use rust_decimal::Decimal;
use std::path::Path;
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingConfig {
    pub symbol: String,
    pub private_key: String,
    pub max_position: Decimal,
    pub max_notional: Decimal,
    pub target_spread_pct: f64,
    pub order_size: Decimal,
    pub risk_aversion: f64,
    pub inventory_target: f64,
    pub vol_window_secs: u64,
    pub throttle_ms: u64,
    pub post_only: bool,
    pub max_order_age_secs: u64,
    pub min_balance: Decimal,
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            symbol: "BTC-USD".to_string(),
            private_key: "YOUR_PRIVATE_KEY_HERE".to_string(),
            max_position: Decimal::from(1000), // 1000 USD
            max_notional: Decimal::from(50),   // 50 USD per order
            target_spread_pct: 0.0005,        // 0.05% spread (ultra-tight)
            order_size: Decimal::from(10),     // 10 USD orders
            risk_aversion: 0.001,              // Very low risk
            inventory_target: 0.0,             // Stay neutral
            vol_window_secs: 60,               // 1-minute volatility window
            throttle_ms: 10,                   // 10ms between orders (ultra-fast)
            post_only: false,                  // Allow taker orders
            max_order_age_secs: 30,            // Cancel orders after 30s
            min_balance: Decimal::from(100),   // Minimum balance required
        }
    }
}

impl TradingConfig {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: TradingConfig = serde_json::from_str(&content)?;
        Ok(config)
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if self.private_key == "YOUR_PRIVATE_KEY_HERE" {
            anyhow::bail!("Private key not configured");
        }
        
        if self.target_spread_pct <= 0.0 {
            anyhow::bail!("Target spread must be positive");
        }
        
        if self.order_size <= Decimal::ZERO {
            anyhow::bail!("Order size must be positive");
        }
        
        if self.throttle_ms < 1 {
            anyhow::bail!("Throttle must be at least 1ms");
        }
        
        Ok(())
    }
}
