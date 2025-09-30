# Hyperliquid Trading Bot

A high-frequency market making bot for the Hyperliquid exchange, supporting both Python and Rust implementations.

## Features

- **Market Making Strategy**: Avellaneda-Stoikov algorithm for optimal bid/ask pricing
- **Risk Management**: Position limits, drawdown protection, order rate limiting
- **Multi-Exchange Support**: Hyperliquid integration with official SDK
- **Real-time Order Book**: L2 cache with efficient data structures
- **Live Trading**: Real order placement with safety checks
- **Paper Trading**: Risk-free testing environment

## Architecture

### Python Implementation
- **Core Bot**: `main_trading_bot.py` - Main trading bot
- **Exchange Adapter**: `mm_bot/exchanges/hyperliquid_official.py` - Hyperliquid integration
- **Strategy**: `mm_bot/strategy/avellaneda_stoikov.py` - Market making algorithm
- **Risk Management**: `mm_bot/risk/` - Position and drawdown limits
- **Order Management**: `mm_bot/execution/order_manager.py` - Order lifecycle

### Rust Implementation
- **Core Bot**: `src/bin/live_trading.rs` - Main trading bot
- **Paper Trading**: `src/bin/paper_trading.rs` - Testing environment
- **Rust Bot**: `src/bin/rust_trading_bot.rs` - Alternative Rust implementation
- **Working Bot**: `src/bin/working_rust_bot.rs` - Simple working Rust bot
- **Strategy**: `src/strategy.rs` - Market making algorithm
- **Risk Management**: `src/risk.rs` - Position and drawdown limits

## Quick Start

### Python Setup

1. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

2. **Configure Trading**:
   ```bash
   cp configs/hyperliquid_safe.yaml configs/my_config.yaml
   # Edit configs/my_config.yaml with your API credentials
   ```

3. **Run Paper Trading**:
   ```bash
   python main_trading_bot.py
   ```

### Rust Setup

1. **Install Dependencies**:
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

4. **Run Working Rust Bot**:
   ```bash
   # Working Rust trading bot (simulation mode)
   cargo run --bin working_rust_bot
   ```

## Configuration

### Key Parameters

```yaml
# Trading Configuration
symbols:
  - "FTT"  # FTX Token

# Risk Management
risk:
  max_position: 22.0        # Max position size
  max_notional: 75.0        # Max notional value
  max_drawdown_pct: 7.7     # Stop loss percentage

# Order Management
quote_size_base: 11.0       # Order size
min_spread_bps: 10          # Minimum spread (0.1%)
max_spread_bps: 50          # Maximum spread (0.5%)
```

### API Credentials

**⚠️ IMPORTANT: Never commit real API credentials to Git!**

1. **Copy the template config**:
   ```bash
   cp configs/hyperliquid_template.yaml configs/hyperliquid_safe.yaml
   ```

2. **Add your credentials** to `configs/hyperliquid_safe.yaml`:
   ```yaml
   exchanges:
     hyperliquid:
       api_key: "your_actual_api_key"
       api_secret: "your_actual_api_secret"
   ```

3. **For Rust bots**, update the credentials in the source files:
   ```rust
   let private_key = "your_actual_private_key";
   let profile_address = "your_actual_profile_address";
   ```

4. **Add to .gitignore** to prevent accidental commits:
   ```
   configs/hyperliquid_safe.yaml
   configs/*_private.yaml
   configs/*_personal.yaml
   ```

## Safety Features

- **Position Limits**: Maximum position size and notional value
- **Drawdown Protection**: Automatic stop-loss at configured percentage
- **Order Rate Limiting**: Prevents excessive order placement
- **Balance Checks**: Ensures sufficient funds before trading
- **Emergency Stop**: Cancels all orders and closes positions

## Risk Warning

⚠️ **This bot trades with real money on live markets. Use at your own risk.**

- Start with paper trading to test strategies
- Use small position sizes initially
- Monitor the bot closely during live trading
- Set appropriate risk limits
- Never risk more than you can afford to lose

## Development

### Project Structure

```
├── mm_bot/                    # Python trading framework
│   ├── exchanges/             # Exchange adapters
│   ├── strategy/              # Trading strategies
│   ├── risk/                  # Risk management
│   ├── execution/             # Order management
│   └── marketdata/            # Market data handling
├── src/                       # Rust implementation
│   ├── bin/                   # Binary executables
│   ├── strategy.rs            # Trading strategy
│   └── risk.rs                # Risk management
├── configs/                   # Configuration files
└── main_trading_bot.py        # Main Python bot
```

### Adding New Exchanges

1. Create adapter in `mm_bot/exchanges/`
2. Implement required methods from `base.py`
3. Add configuration in `configs/`

### Adding New Strategies

1. Create strategy in `mm_bot/strategy/`
2. Implement `calculate_quotes()` method
3. Add configuration parameters

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the configuration examples