#!/usr/bin/env rust
/*
 * Paper Trading Bot - Test without real money
 * 
 * Simulates market making with virtual money to test strategies
 */

use anyhow::Result;
use hyperliquid_trading_bot::config::TradingConfig;
use hyperliquid_trading_bot::paper_trading::PaperTradingBot;
use rust_decimal::Decimal;
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("🧪==============================================================================🧪");
    println!("🎯                    HYPERLIQUID PAPER TRADING BOT                    🎯");
    println!("💰==============================================================================💰");
    println!();
    println!("⚠️  THIS IS A SIMULATION - NO REAL MONEY AT RISK!");
    println!("📊 Testing market making strategy with virtual $10,000");
    println!("🐧 Symbol: PENGU-USD (simulated)");
    println!("📏 Target Spread: 0.1% (tight for PENGU)");
    println!("⚡ Order Refresh: 10ms (ultra-fast)");
    println!();

    // Load configuration for PENGU
    let config = TradingConfig {
        symbol: "PENGU-USD".to_string(),
        private_key: "YOUR_PRIVATE_KEY_HERE".to_string(),
        max_position: Decimal::from(1000),
        max_notional: Decimal::from(50),
        target_spread_pct: 0.001,  // 0.1% spread for PENGU
        order_size: Decimal::from(10),
        risk_aversion: 0.001,
        inventory_target: 0.0,
        vol_window_secs: 60,
        throttle_ms: 10,
        post_only: false,
        max_order_age_secs: 30,
        min_balance: Decimal::from(100),
    };
    
    // Create paper trading bot with $10,000 virtual balance
    let mut bot = PaperTradingBot::new(config, Decimal::from(10000));
    
    // Handle Ctrl+C gracefully
    let bot_handle = tokio::spawn(async move {
        tokio::signal::ctrl_c().await.expect("Failed to listen for ctrl+c");
        println!("\n🛑 Stopping paper trading bot...");
        std::process::exit(0);
    });
    
    // Start paper trading
    let trading_handle = tokio::spawn(async move {
        if let Err(e) = bot.start().await {
            eprintln!("❌ Error running paper trading bot: {}", e);
        }
    });
    
    // Wait for either signal or completion
    tokio::select! {
        _ = bot_handle => {
            println!("🛑 Bot stopped by user");
        }
        _ = trading_handle => {
            println!("✅ Paper trading session completed");
        }
    }
    
    Ok(())
}
