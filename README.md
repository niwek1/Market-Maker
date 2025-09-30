# 🚀 Hyperliquid Market Making Bot

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://rust-lang.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-niwek1%2FMarket--Maker-black.svg)](https://github.com/niwek1/Market-Maker)

A professional-grade, high-frequency market making bot for the **Hyperliquid** exchange, featuring both **Python** and **Rust** implementations with advanced risk management and real-time order execution.

## ✨ Features

### 🎯 **Core Trading Features**
- **Avellaneda-Stoikov Algorithm**: Sophisticated market making strategy for optimal bid/ask pricing
- **Real-time Order Management**: Live order placement, modification, and cancellation
- **Multi-Asset Support**: Trade FTT, GMT, PENGU, and other Hyperliquid assets
- **High-Frequency Trading**: Ultra-fast order execution with configurable throttling

### 🛡️ **Risk Management**
- **Position Limits**: Maximum position size and notional value controls
- **Drawdown Protection**: Automatic stop-loss at configured percentage
- **Order Rate Limiting**: Prevents excessive order placement
- **Balance Monitoring**: Real-time account balance and margin checks
- **Emergency Stop**: Instant cancellation of all orders and position closure

### 🔧 **Technical Features**
- **Dual Implementation**: Both Python and Rust for different use cases
- **Paper Trading**: Risk-free testing environment with virtual money
- **Live Trading**: Real money trading with comprehensive safety checks
- **Real-time Monitoring**: Live dashboard with P&L tracking and metrics
- **Configurable Strategy**: Customizable parameters for different market conditions

## 🏗️ Architecture

### 🐍 **Python Implementation**
```
main_trading_bot.py              # Main trading bot with live dashboard
├── mm_bot/
│   ├── exchanges/
│   │   └── hyperliquid_official.py  # Hyperliquid SDK integration
│   ├── strategy/
│   │   └── avellaneda_stoikov.py    # Market making algorithm
│   ├── risk/
│   │   ├── limits.py                # Position and drawdown limits
│   │   ├── kill_switch.py           # Emergency stop functionality
│   │   └── inventory.py             # Inventory management
│   ├── execution/
│   │   └── order_manager.py         # Order lifecycle management
│   └── marketdata/
│       └── l2_cache.py              # Real-time order book cache
```

### 🦀 **Rust Implementation**
```
src/
├── bin/
│   ├── live_trading.rs          # Main live trading bot
│   ├── paper_trading.rs         # Paper trading simulation
│   ├── rust_trading_bot.rs      # Alternative implementation
│   └── working_rust_bot.rs      # Simple working example
├── strategy.rs                  # Trading strategy implementation
├── risk.rs                      # Risk management system
└── order_book.rs                # Order book data structures
```

## 🚀 Quick Start

### 📋 **Prerequisites**
- **Python 3.8+** with pip
- **Rust 1.70+** with Cargo
- **Hyperliquid Account** with API access
- **Minimum $50** for safe trading (recommended $100+)

### 🐍 **Python Setup**

1. **Clone and Install**:
   ```bash
   git clone https://github.com/niwek1/Market-Maker.git
   cd Market-Maker
   pip install -e .
   ```

2. **Configure Trading**:
   ```bash
   cp configs/hyperliquid_template.yaml configs/hyperliquid_safe.yaml
   # Edit configs/hyperliquid_safe.yaml with your API credentials
   ```

3. **Start Paper Trading** (Recommended First):
   ```bash
   python main_trading_bot.py
   ```

4. **Start Live Trading** (After Testing):
   ```bash
   # Update config to live mode
   python main_trading_bot.py
   ```

### 🦀 **Rust Setup**

1. **Build the Project**:
   ```bash
   cargo build --release
   ```

2. **Run Paper Trading**:
   ```bash
   cargo run --bin paper_trading
   ```

3. **Run Live Trading**:
   ```bash
   cargo run --bin live_trading
   ```

4. **Run Simple Bot**:
   ```bash
   cargo run --bin working_rust_bot
   ```

## ⚙️ Configuration

### 🔑 **API Credentials Setup**

**⚠️ CRITICAL: Never commit real API credentials to Git!**

1. **Copy Template Config**:
   ```bash
   cp configs/hyperliquid_template.yaml configs/hyperliquid_safe.yaml
   ```

2. **Add Your Credentials** to `configs/hyperliquid_safe.yaml`:
   ```yaml
   exchanges:
     hyperliquid:
       api_key: "your_actual_api_key"
       api_secret: "your_actual_api_secret"
   ```

3. **For Rust Bots**, update credentials in source files:
   ```rust
   let private_key = "your_actual_private_key";
   let profile_address = "your_actual_profile_address";
   ```

### 📊 **Trading Configuration**

```yaml
# Asset Selection
symbols:
  - "FTT"        # FTX Token (high liquidity)
  # - "GMT"      # Green Metaverse Token
  # - "PENGU"    # PENGU Token

# Risk Management
risk:
  max_position: 22.0         # Max 22 tokens per position
  max_notional: 75.0         # Max $75 notional value
  max_drawdown_pct: 7.7      # Stop at 7.7% loss

# Order Management
quote_size_base: 11.0        # 11 tokens per order (~$10)
min_spread_bps: 10           # 0.1% minimum spread
max_spread_bps: 50           # 0.5% maximum spread
target_spread_pct: 0.1       # 0.1% target spread

# Performance
throttle_ms: 10              # 10ms between orders (ultra-fast)
replace_after_ms: 500        # Replace orders every 0.5s
max_order_rate_per_min: 100  # Max 100 orders per minute
```

## 🛡️ Safety Features

### 🚨 **Risk Controls**
- **Position Limits**: Maximum position size and notional value
- **Drawdown Protection**: Automatic stop-loss at configured percentage
- **Order Rate Limiting**: Prevents excessive order placement
- **Balance Checks**: Ensures sufficient funds before trading
- **Margin Monitoring**: Real-time margin usage tracking

### 🛑 **Emergency Features**
- **Kill Switch**: Instant cancellation of all orders
- **Position Closure**: Automatic position liquidation
- **Balance Alerts**: Notifications for low balance
- **Error Handling**: Graceful handling of API errors

## 📈 **Performance Metrics**

The bot tracks comprehensive metrics including:
- **Orders Placed/Filled**: Order execution statistics
- **Fill Rate**: Percentage of orders that get filled
- **P&L Tracking**: Real-time profit and loss
- **Volume**: Total trading volume
- **Latency**: Order execution timing
- **Risk Metrics**: Position and drawdown monitoring

## ⚠️ Risk Warning

**🚨 IMPORTANT: This bot trades with real money on live markets!**

### 📋 **Before Trading**
- ✅ **Start with paper trading** to test strategies
- ✅ **Use small position sizes** initially
- ✅ **Monitor the bot closely** during live trading
- ✅ **Set appropriate risk limits** for your account size
- ✅ **Never risk more than you can afford to lose**

### 💰 **Recommended Account Sizes**
- **Minimum**: $50 (for testing)
- **Recommended**: $100+ (for stable operation)
- **Optimal**: $500+ (for better performance)

## 🔧 Development

### 📁 **Project Structure**
```
Market-Maker/
├── main_trading_bot.py           # 🐍 Main Python trading bot
├── configs/                      # 📋 Configuration files
│   ├── hyperliquid_template.yaml # Template (safe to commit)
│   └── hyperliquid_safe.yaml     # Your config (gitignored)
├── mm_bot/                       # 🐍 Python trading framework
│   ├── exchanges/                # Exchange integrations
│   ├── strategy/                 # Trading strategies
│   ├── risk/                     # Risk management
│   ├── execution/                # Order management
│   └── marketdata/               # Market data handling
├── src/                          # 🦀 Rust implementation
│   ├── bin/                      # Binary executables
│   ├── strategy.rs               # Trading strategy
│   └── risk.rs                   # Risk management
├── Cargo.toml                    # 🦀 Rust dependencies
├── pyproject.toml                # 🐍 Python dependencies
└── README.md                     # 📖 This file
```

### 🔌 **Adding New Exchanges**
1. Create adapter in `mm_bot/exchanges/`
2. Implement required methods from `base.py`
3. Add configuration in `configs/`
4. Update documentation

### 📊 **Adding New Strategies**
1. Create strategy in `mm_bot/strategy/`
2. Implement `calculate_quotes()` method
3. Add configuration parameters
4. Test with paper trading

## 📚 **Documentation**

- **Setup Guide**: See Quick Start section above
- **Configuration**: See Configuration section
- **API Reference**: Check individual module docstrings
- **Examples**: See `configs/hyperliquid_template.yaml`

## 🤝 **Contributing**

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### 📝 **Development Guidelines**
- Follow existing code style
- Add tests for new features
- Update documentation
- Test with paper trading first

## 📄 **License**

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🆘 **Support**

### 🐛 **Issues**
- **GitHub Issues**: [Create an issue](https://github.com/niwek1/Market-Maker/issues)
- **Documentation**: Check this README and code comments
- **Configuration**: Review `configs/hyperliquid_template.yaml`

### 💬 **Community**
- **Discussions**: Use GitHub Discussions for questions
- **Pull Requests**: Submit improvements and bug fixes
- **Wiki**: Check the project wiki for additional resources

## 🙏 **Acknowledgments**

- **Hyperliquid** for providing the trading platform
- **Avellaneda & Stoikov** for the market making algorithm
- **Open Source Community** for various libraries and tools

---

**⚠️ Disclaimer**: This software is for educational and research purposes. Trading cryptocurrencies involves substantial risk of loss. The authors are not responsible for any financial losses. Use at your own risk.

**🔗 Repository**: [https://github.com/niwek1/Market-Maker](https://github.com/niwek1/Market-Maker)

**⭐ Star this repository if you find it helpful!**