use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: Decimal,
    pub size: Decimal,
}

#[derive(Debug, Clone)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub timestamp: DateTime<Utc>,
    pub sequence: u64,
}

impl OrderBook {
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            bids: Vec::new(),
            asks: Vec::new(),
            timestamp: Utc::now(),
            sequence: 0,
        }
    }

    pub fn update(&mut self, bids: Vec<OrderBookLevel>, asks: Vec<OrderBookLevel>) {
        self.bids = bids;
        self.asks = asks;
        self.timestamp = Utc::now();
        self.sequence += 1;
    }

    pub fn mid_price(&self) -> Option<Decimal> {
        if let (Some(best_bid), Some(best_ask)) = (self.best_bid(), self.best_ask()) {
            Some((best_bid.price + best_ask.price) / Decimal::from(2))
        } else {
            None
        }
    }

    pub fn best_bid(&self) -> Option<&OrderBookLevel> {
        self.bids.first()
    }

    pub fn best_ask(&self) -> Option<&OrderBookLevel> {
        self.asks.first()
    }

    pub fn spread(&self) -> Option<Decimal> {
        if let (Some(bid), Some(ask)) = (self.best_bid(), self.best_ask()) {
            Some(ask.price - bid.price)
        } else {
            None
        }
    }

    pub fn spread_pct(&self) -> Option<f64> {
        if let (Some(bid), Some(ask)) = (self.best_bid(), self.best_ask()) {
            let spread = ask.price - bid.price;
            let mid = (bid.price + ask.price) / Decimal::from(2);
            Some((spread / mid).to_f64().unwrap_or(0.0) * 100.0)
        } else {
            None
        }
    }

    pub fn volume_at_price(&self, price: Decimal, side: OrderSide) -> Decimal {
        let levels = match side {
            OrderSide::Buy => &self.bids,
            OrderSide::Sell => &self.asks,
        };

        levels.iter()
            .filter(|level| level.price == price)
            .map(|level| level.size)
            .sum()
    }

    pub fn total_volume(&self, side: OrderSide) -> Decimal {
        let levels = match side {
            OrderSide::Buy => &self.bids,
            OrderSide::Sell => &self.asks,
        };

        levels.iter().map(|level| level.size).sum()
    }

    pub fn depth_at_levels(&self, levels: usize, side: OrderSide) -> Vec<OrderBookLevel> {
        let source = match side {
            OrderSide::Buy => &self.bids,
            OrderSide::Sell => &self.asks,
        };

        source.iter().take(levels).cloned().collect()
    }

    pub fn is_valid(&self) -> bool {
        !self.bids.is_empty() && !self.asks.is_empty()
    }

    pub fn age_seconds(&self) -> f64 {
        let now = Utc::now();
        (now - self.timestamp).num_milliseconds() as f64 / 1000.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone)]
pub struct MicrostructureFeatures {
    pub mid_price: Decimal,
    pub spread: Decimal,
    pub spread_pct: f64,
    pub bid_volume: Decimal,
    pub ask_volume: Decimal,
    pub volume_imbalance: f64,
    pub microprice: Decimal,
    pub volatility: f64,
    pub timestamp: DateTime<Utc>,
}

impl MicrostructureFeatures {
    pub fn from_order_book(order_book: &OrderBook, price_history: &[(DateTime<Utc>, Decimal)]) -> Self {
        let mid_price = order_book.mid_price().unwrap_or_default();
        let spread = order_book.spread().unwrap_or_default();
        let spread_pct = order_book.spread_pct().unwrap_or(0.0);
        
        let bid_volume = order_book.total_volume(OrderSide::Buy);
        let ask_volume = order_book.total_volume(OrderSide::Sell);
        
        let volume_imbalance = if bid_volume + ask_volume > Decimal::ZERO {
            ((bid_volume - ask_volume) / (bid_volume + ask_volume)).to_f64().unwrap_or(0.0)
        } else {
            0.0
        };

        // Calculate microprice (volume-weighted mid)
        let microprice = if let (Some(bid), Some(ask)) = (order_book.best_bid(), order_book.best_ask()) {
            let total_size = bid.size + ask.size;
            if total_size > Decimal::ZERO {
                (bid.price * ask.size + ask.price * bid.size) / total_size
            } else {
                mid_price
            }
        } else {
            mid_price
        };

        // Calculate short-term volatility
        let volatility = Self::calculate_volatility(price_history, 10);

        Self {
            mid_price,
            spread,
            spread_pct,
            bid_volume,
            ask_volume,
            volume_imbalance,
            microprice,
            volatility,
            timestamp: Utc::now(),
        }
    }

    fn calculate_volatility(price_history: &[(DateTime<Utc>, Decimal)], window: usize) -> f64 {
        if price_history.len() < 2 {
            return 0.0;
        }

        let recent_prices: Vec<f64> = price_history
            .iter()
            .rev()
            .take(window)
            .map(|(_, price)| price.to_f64().unwrap_or(0.0))
            .collect();

        if recent_prices.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = recent_prices
            .windows(2)
            .map(|window| (window[1] / window[0]).ln())
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;

        variance.sqrt()
    }
}
