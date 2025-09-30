use hyperliquid_trading_bot::{
    config::TradingConfig,
    order_book::{OrderBook, OrderSide},
    strategy::AvellanedaStoikovStrategy,
    risk::RiskManager,
    utils::{format_currency, calculate_spread_pct},
};
use hyperliquid_rust_sdk::Exchange;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::interval;
use log::{debug, error, info};
use anyhow::Result;
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone)]
pub struct Order {
    pub id: Uuid,
    pub symbol: String,
    pub side: OrderSide,
    pub size: Decimal,
    pub price: Decimal,
    pub timestamp: DateTime<Utc>,
    pub status: String,
}

pub struct LiveTradingBot {
    config: TradingConfig,
    exchange: Arc<Exchange>,
    order_book: Arc<RwLock<OrderBook>>,
    strategy: Arc<RwLock<AvellanedaStoikovStrategy>>,
    risk_manager: Arc<RwLock<RiskManager>>,
    active_orders: Arc<RwLock<HashMap<Uuid, Order>>>,
    running: Arc<RwLock<bool>>,
}

impl LiveTradingBot {
    pub fn new(config: TradingConfig) -> Result<Self> {
        // Initialize Hyperliquid exchange
        let exchange = Arc::new(Exchange::new(
            &config.private_key,
            false, // mainnet
        ));

        // Initialize order book
        let order_book = Arc::new(RwLock::new(OrderBook::new(
            config.symbol.clone(),
            20, // max_levels
            200, // feature_history_size
        )));

        // Initialize strategy
        let strategy_params = hyperliquid_trading_bot::strategy::AvellanedaStoikovParams {
            gamma: Decimal::from_f64_retain(0.0001).unwrap(),
            kappa: Decimal::from_f64_retain(0.001).unwrap(),
            risk_aversion: Decimal::from_f64_retain(config.risk_aversion).unwrap(),
            inventory_target: Decimal::from_f64_retain(config.inventory_target).unwrap(),
            inventory_band: Decimal::from_f64_retain(0.00001).unwrap(),
            vol_window_secs: config.vol_window_secs,
            vol_floor: Decimal::from_f64_retain(0.00001).unwrap(),
            vol_scaling: Decimal::from_f64_retain(0.01).unwrap(),
            target_spread_pct: Decimal::from_f64_retain(config.target_spread_pct).unwrap(),
            imbalance_window_ms: 100,
            microprice_alpha: Decimal::from_f64_retain(0.7).unwrap(),
        };
        let strategy = Arc::new(RwLock::new(AvellanedaStoikovStrategy::new(strategy_params)));

        // Initialize risk manager
        let risk_limits = hyperliquid_trading_bot::risk::RiskLimits {
            max_position: config.max_position,
            max_notional: config.max_notional,
            max_drawdown_pct: Decimal::from_f64_retain(10.0).unwrap(),
            max_order_rate_per_min: 120, // Increased for live trading
            min_spread_bps: Decimal::from_f64_retain(1.0).unwrap(),
            max_spread_bps: Decimal::from_f64_retain(100.0).unwrap(),
        };
        let risk_manager = Arc::new(RwLock::new(RiskManager::new(risk_limits, Decimal::from(10000))));

        Ok(LiveTradingBot {
            config,
            exchange,
            order_book,
            strategy,
            risk_manager,
            active_orders: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(RwLock::new(true)),
        })
    }

    pub async fn run(&self) -> Result<()> {
        info!("🚀 Starting Live Trading Bot for {}", self.config.symbol);
        info!("💰 Max Position: ${}", self.config.max_position);
        info!("📏 Target Spread: {:.3}%", self.config.target_spread_pct);
        info!("⚡ Throttle: {}ms", self.config.throttle_ms);

        // Start order book updates
        let order_book_task = {
            let exchange = self.exchange.clone();
            let order_book = self.order_book.clone();
            let symbol = self.config.symbol.clone();
            let running = self.running.clone();
            
            tokio::spawn(async move {
                let mut interval = interval(Duration::from_millis(100)); // 10 updates per second
                
                while *running.read().await {
                    interval.tick().await;
                    
                    match exchange.get_l2_book(&symbol).await {
                        Ok(l2_book) => {
                            let mut book = order_book.write().await;
                            book.update_order_book(l2_book);
                        }
                        Err(e) => {
                            error!("Failed to fetch order book: {:?}", e);
                        }
                    }
                }
            })
        };

        // Start trading loop
        let trading_task = {
            let config = self.config.clone();
            let exchange = self.exchange.clone();
            let order_book = self.order_book.clone();
            let strategy = self.strategy.clone();
            let risk_manager = self.risk_manager.clone();
            let active_orders = self.active_orders.clone();
            let running = self.running.clone();
            
            tokio::spawn(async move {
                let mut interval = interval(Duration::from_millis(config.throttle_ms));
                
                while *running.read().await {
                    interval.tick().await;
                    
                    // Get current market data
                    let book = order_book.read().await;
                    let features = book.get_features();
                    let mid_price = features.mid;
                    
                    if mid_price.is_zero() {
                        debug!("Mid price is zero, skipping trading cycle");
                        continue;
                    }

                    // Calculate optimal quotes
                    let strategy_guard = strategy.read().await;
                    let (bid_price, ask_price) = strategy_guard.get_optimal_quotes(
                        mid_price,
                        book.get_current_imbalance(),
                        book.get_current_volatility(),
                        Decimal::ZERO, // current_inventory
                    );
                    drop(strategy_guard);

                    // Check risk limits
                    let risk_guard = risk_manager.read().await;
                    if let Err(e) = risk_guard.check_order_risk(
                        OrderSide::Buy,
                        config.order_size,
                        bid_price,
                    ) {
                        debug!("Risk check failed for buy order: {}", e);
                        continue;
                    }
                    if let Err(e) = risk_guard.check_order_risk(
                        OrderSide::Sell,
                        config.order_size,
                        ask_price,
                    ) {
                        debug!("Risk check failed for sell order: {}", e);
                        continue;
                    }
                    drop(risk_guard);

                    // Place orders
                    self.place_orders(&exchange, &active_orders, bid_price, ask_price).await;
                }
            })
        };

        // Start display updates
        let display_task = {
            let order_book = self.order_book.clone();
            let active_orders = self.active_orders.clone();
            let running = self.running.clone();
            let symbol = self.config.symbol.clone();
            
            tokio::spawn(async move {
                let mut interval = interval(Duration::from_secs(1)); // Update every second
                
                while *running.read().await {
                    interval.tick().await;
                    
                    let book = order_book.read().await;
                    let features = book.get_features();
                    let orders = active_orders.read().await;
                    
                    println!("\x1B[2J\x1B[1;1H"); // Clear screen
                    println!("🚀==============================================================================🚀");
                    println!("🎯                         LIVE TRADING BOT - {}                         🎯", symbol);
                    println!("💰==============================================================================💰");
                    println!();
                    println!("📊 MARKET DATA:");
                    println!("   💎 {} Price: ${:.6}", symbol, features.mid);
                    println!("   📏 Current Spread: {:.3}%", calculate_spread_pct(features.spread, features.mid));
                    println!("   🎯 Target Spread: {:.3}%", config.target_spread_pct);
                    println!();
                    println!("📈 TRADING STATUS:");
                    println!("   🎯 Active Orders: {}", orders.len());
                    println!("   ⚡ Status: 🟢 ACTIVE");
                    println!("   ⏰ Time: {}", Utc::now().format("%H:%M:%S"));
                    println!();
                    println!("Press Ctrl+C to stop");
                }
            })
        };

        // Wait for all tasks
        tokio::try_join!(order_book_task, trading_task, display_task)?;

        Ok(())
    }

    async fn place_orders(
        &self,
        exchange: &Exchange,
        active_orders: &Arc<RwLock<HashMap<Uuid, Order>>>,
        bid_price: Decimal,
        ask_price: Decimal,
    ) {
        // Place buy order
        match exchange.order(
            &self.config.symbol,
            true, // is_buy
            self.config.order_size,
            bid_price,
            false, // post_only
        ).await {
            Ok(order_id) => {
                let order = Order {
                    id: Uuid::new_v4(),
                    symbol: self.config.symbol.clone(),
                    side: OrderSide::Buy,
                    size: self.config.order_size,
                    price: bid_price,
                    timestamp: Utc::now(),
                    status: "Pending".to_string(),
                };
                
                let mut orders = active_orders.write().await;
                orders.insert(order.id, order);
                
                info!("Placed BUY order: {} @ ${:.6}", self.config.order_size, bid_price);
            }
            Err(e) => {
                error!("Failed to place buy order: {:?}", e);
            }
        }

        // Place sell order
        match exchange.order(
            &self.config.symbol,
            false, // is_buy
            self.config.order_size,
            ask_price,
            false, // post_only
        ).await {
            Ok(order_id) => {
                let order = Order {
                    id: Uuid::new_v4(),
                    symbol: self.config.symbol.clone(),
                    side: OrderSide::Sell,
                    size: self.config.order_size,
                    price: ask_price,
                    timestamp: Utc::now(),
                    status: "Pending".to_string(),
                };
                
                let mut orders = active_orders.write().await;
                orders.insert(order.id, order);
                
                info!("Placed SELL order: {} @ ${:.6}", self.config.order_size, ask_price);
            }
            Err(e) => {
                error!("Failed to place sell order: {:?}", e);
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    // Load configuration for PENGU live trading
    let config = TradingConfig {
        symbol: "PENGU-USD".to_string(),
        private_key: "YOUR_PRIVATE_KEY_HERE".to_string(), // Replace with your actual private key
        max_position: Decimal::from(1000), // $1000 max position
        max_notional: Decimal::from(50),   // $50 per order
        target_spread_pct: 0.001,         // 0.1% spread for PENGU
        order_size: Decimal::from(10),    // 10 PENGU per order
        risk_aversion: 0.001,
        inventory_target: 0.0,
        vol_window_secs: 60,
        throttle_ms: 100, // 100ms throttle for live trading
    };

    // Create and run bot
    let bot = LiveTradingBot::new(config)?;
    bot.run().await?;

    Ok(())
}
