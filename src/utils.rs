use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use chrono::{DateTime, Utc};

pub fn format_currency(amount: Decimal) -> String {
    format!("${:.2}", amount)
}

pub fn format_percentage(value: f64) -> String {
    format!("{:.4}%", value)
}

pub fn format_timestamp(dt: DateTime<Utc>) -> String {
    dt.format("%H:%M:%S%.3f").to_string()
}

pub fn calculate_pnl(entry_price: Decimal, current_price: Decimal, size: Decimal, side: OrderSide) -> Decimal {
    match side {
        OrderSide::Buy => (current_price - entry_price) * size,
        OrderSide::Sell => (entry_price - current_price) * size,
    }
}

pub fn calculate_fill_rate(orders_placed: u64, orders_filled: u64) -> f64 {
    if orders_placed == 0 {
        0.0
    } else {
        (orders_filled as f64 / orders_placed as f64) * 100.0
    }
}

pub fn calculate_volume_weighted_price(prices: &[(Decimal, Decimal)]) -> Option<Decimal> {
    if prices.is_empty() {
        return None;
    }

    let total_volume: Decimal = prices.iter().map(|(_, size)| size).sum();
    if total_volume == Decimal::ZERO {
        return None;
    }

    let weighted_sum: Decimal = prices.iter()
        .map(|(price, size)| price * size)
        .sum();

    Some(weighted_sum / total_volume)
}

pub fn calculate_volatility(prices: &[Decimal], window: usize) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }

    let recent_prices: Vec<f64> = prices
        .iter()
        .rev()
        .take(window)
        .map(|p| p.to_f64().unwrap_or(0.0))
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

pub fn round_to_tick_size(price: Decimal, tick_size: Decimal) -> Decimal {
    (price / tick_size).round() * tick_size
}

pub fn calculate_spread_pct(bid: Decimal, ask: Decimal) -> f64 {
    if bid == Decimal::ZERO || ask == Decimal::ZERO {
        return 0.0;
    }
    
    let spread = ask - bid;
    let mid = (bid + ask) / Decimal::from(2);
    (spread / mid).to_f64().unwrap_or(0.0) * 100.0
}

pub fn is_price_valid(price: Decimal) -> bool {
    price > Decimal::ZERO && price < Decimal::from(1_000_000) // Reasonable price range
}

pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

pub fn exponential_moving_average(current: f64, previous: f64, alpha: f64) -> f64 {
    alpha * current + (1.0 - alpha) * previous
}

pub fn calculate_sharpe_ratio(returns: &[f64], risk_free_rate: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let excess_return = mean_return - risk_free_rate;
    
    let variance = returns.iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>() / returns.len() as f64;
    
    let std_dev = variance.sqrt();
    
    if std_dev == 0.0 {
        0.0
    } else {
        excess_return / std_dev
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_fill_rate() {
        assert_eq!(calculate_fill_rate(0, 0), 0.0);
        assert_eq!(calculate_fill_rate(100, 50), 50.0);
        assert_eq!(calculate_fill_rate(100, 100), 100.0);
    }

    #[test]
    fn test_calculate_spread_pct() {
        let bid = Decimal::from(100);
        let ask = Decimal::from(101);
        assert!((calculate_spread_pct(bid, ask) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5, 0, 10), 5);
        assert_eq!(clamp(-1, 0, 10), 0);
        assert_eq!(clamp(15, 0, 10), 10);
    }
}
