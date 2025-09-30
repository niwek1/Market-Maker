"""
Configuration management using Pydantic with YAML and environment variable support.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class ExchangeConfig(BaseModel):
    """Exchange-specific configuration."""
    
    name: str = Field(..., description="Exchange name (binance, kraken, coinbase)")
    api_key: Optional[str] = Field(None, description="API key")
    api_secret: Optional[str] = Field(None, description="API secret")
    passphrase: Optional[str] = Field(None, description="Passphrase (Coinbase only)")
    sandbox: bool = Field(True, description="Use sandbox/testnet")
    rate_limit: int = Field(10, description="Requests per second limit")
    timeout: int = Field(30, description="Request timeout in seconds")


class FeeConfig(BaseModel):
    """Fee configuration per exchange."""
    
    maker_bps: float = Field(5.0, description="Maker fee in basis points")
    taker_bps: float = Field(10.0, description="Taker fee in basis points")
    funding_rate: Optional[float] = Field(None, description="Funding rate for perps")


class RiskConfig(BaseModel):
    """Risk management configuration."""
    
    max_position: float = Field(1.0, description="Maximum position size (base units)")
    max_notional: float = Field(10000.0, description="Maximum notional exposure")
    max_drawdown_pct: float = Field(5.0, description="Maximum drawdown percentage")
    max_order_rate_per_min: int = Field(60, description="Maximum orders per minute")
    quote_size_base: float = Field(0.01, description="Quote size in base units")
    min_spread_bps: float = Field(5.0, description="Minimum spread in basis points")
    max_spread_bps: float = Field(500.0, description="Maximum spread in basis points")
    position_limit_buffer: float = Field(0.1, description="Buffer before hitting limits")
    
    @validator("max_drawdown_pct")
    def validate_drawdown(cls, v: float) -> float:
        if not 0 < v <= 100:
            raise ValueError("max_drawdown_pct must be between 0 and 100")
        return v
    
    @validator("min_spread_bps", "max_spread_bps")
    def validate_spread(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Spread must be non-negative")
        return v


class AvellanedaStoikovConfig(BaseModel):
    """Avellaneda-Stoikov strategy parameters."""
    
    gamma: float = Field(0.1, description="Risk aversion parameter")
    kappa: float = Field(1.5, description="Inventory penalty parameter")
    risk_aversion: float = Field(0.1, description="Overall risk aversion")
    inventory_target: float = Field(0.0, description="Target inventory level")
    inventory_band: float = Field(0.5, description="Inventory band around target")
    vol_window_secs: int = Field(300, description="Volatility estimation window")
    imbalance_window_ms: int = Field(1000, description="Order book imbalance window")
    vol_lookback_periods: int = Field(100, description="Volatility lookback periods")
    
    # Spread override for low-priced tokens
    target_spread_pct: float = Field(0.002, description="Target spread percentage (0.002 = 0.2%)")
    vol_floor: float = Field(0.01, description="Minimum volatility floor")
    vol_scaling: float = Field(1.0, description="Volatility scaling factor")
    microprice_alpha: float = Field(0.5, description="Microprice weighting")
    
    # Bandit parameters
    bandit_enabled: bool = Field(False, description="Enable bandit optimization")
    bandit_type: str = Field("ucb", description="Bandit type (ucb, thompson)")
    spread_multiplier_bounds: tuple[float, float] = Field(
        (0.5, 2.0), description="Spread multiplier bounds for bandit"
    )
    bandit_exploration: float = Field(0.1, description="Exploration parameter")
    
    @validator("gamma", "kappa", "risk_aversion")
    def validate_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Parameter must be positive")
        return v


class ExecutionConfig(BaseModel):
    """Order execution configuration."""
    
    post_only: bool = Field(True, description="Use post-only orders")
    time_in_force: str = Field("GTC", description="Time in force")
    replace_after_ms: int = Field(5000, description="Replace orders after milliseconds")
    throttle_ms: int = Field(100, description="Throttle between operations")
    min_quote_lifetime_ms: int = Field(500, description="Minimum quote lifetime")
    max_order_retries: int = Field(3, description="Maximum order retry attempts")
    order_timeout_ms: int = Field(10000, description="Order timeout")
    
    @validator("replace_after_ms", "throttle_ms", "min_quote_lifetime_ms")
    def validate_timing(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Timing parameters must be non-negative")
        return v


class HedgeConfig(BaseModel):
    """Hedging configuration."""
    
    enabled: bool = Field(False, description="Enable hedging")
    hedge_symbol: Optional[str] = Field(None, description="Symbol to hedge with")
    hedge_threshold: float = Field(0.1, description="Threshold to trigger hedge")
    hedge_size_ratio: float = Field(1.0, description="Hedge size ratio")
    hedge_exchange: Optional[str] = Field(None, description="Exchange for hedging")


class StrategyConfig(BaseModel):
    """Overall strategy configuration."""
    
    as_: AvellanedaStoikovConfig = Field(
        default_factory=AvellanedaStoikovConfig,
        alias="as",
        description="Avellaneda-Stoikov parameters"
    )


class DataConfig(BaseModel):
    """Data configuration."""
    
    l2_data_path: str = Field("data/l2", description="L2 data directory path")
    trades_data_path: str = Field("data/trades", description="Trades data directory path")
    timezone: str = Field("UTC", description="Data timezone")
    backtest_start: Optional[str] = Field(None, description="Backtest start time")
    backtest_end: Optional[str] = Field(None, description="Backtest end time")
    data_format: str = Field("parquet", description="Data format (parquet, csv)")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: str = Field("INFO", description="Log level")
    format: str = Field("json", description="Log format (json, text)")
    file_enabled: bool = Field(True, description="Enable file logging")
    file_path: str = Field("logs/mm_bot.log", description="Log file path")
    max_file_size: str = Field("100MB", description="Maximum log file size")
    backup_count: int = Field(5, description="Number of backup log files")


class MetricsConfig(BaseModel):
    """Metrics configuration."""
    
    enabled: bool = Field(True, description="Enable metrics collection")
    port: int = Field(8000, description="Metrics server port")
    path: str = Field("/metrics", description="Metrics endpoint path")
    update_interval: int = Field(10, description="Metrics update interval in seconds")


class Config(BaseSettings):
    """Main configuration class."""
    
    # Core settings
    mode: str = Field("paper", description="Mode: backtest, paper, live")
    exchange: str = Field("binance", description="Primary exchange")
    symbols: List[str] = Field(["BTC/USDT"], description="Trading symbols")
    
    # Component configurations
    exchanges: Dict[str, ExchangeConfig] = Field(
        default_factory=dict, description="Exchange configurations"
    )
    fees: Dict[str, FeeConfig] = Field(
        default_factory=dict, description="Fee configurations"
    )
    risk: RiskConfig = Field(
        default_factory=RiskConfig, description="Risk management settings"
    )
    strategy: StrategyConfig = Field(
        default_factory=StrategyConfig, description="Strategy settings"
    )
    execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig, description="Execution settings"
    )
    hedge: HedgeConfig = Field(
        default_factory=HedgeConfig, description="Hedging settings"
    )
    data: DataConfig = Field(
        default_factory=DataConfig, description="Data settings"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging settings"
    )
    metrics: MetricsConfig = Field(
        default_factory=MetricsConfig, description="Metrics settings"
    )
    
    # Environment overrides
    paper_trading: bool = Field(True, description="Force paper trading mode")
    kill_switch_enabled: bool = Field(True, description="Enable kill switch")
    debug: bool = Field(False, description="Debug mode")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator("mode")
    def validate_mode(cls, v: str) -> str:
        valid_modes = {"backtest", "paper", "live"}
        if v not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}")
        return v
    
    @validator("symbols")
    def validate_symbols(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one symbol must be specified")
        return v
    
    def get_exchange_config(self, exchange_name: str) -> ExchangeConfig:
        """Get configuration for a specific exchange."""
        if exchange_name not in self.exchanges:
            # Create default config
            self.exchanges[exchange_name] = ExchangeConfig(name=exchange_name)
        return self.exchanges[exchange_name]
    
    def get_fee_config(self, exchange_name: str) -> FeeConfig:
        """Get fee configuration for a specific exchange."""
        if exchange_name not in self.fees:
            # Create default config
            self.fees[exchange_name] = FeeConfig()
        return self.fees[exchange_name]
    
    def is_live_mode(self) -> bool:
        """Check if running in live trading mode."""
        return self.mode == "live" and not self.paper_trading
    
    def is_paper_mode(self) -> bool:
        """Check if running in paper trading mode."""
        return self.mode == "paper" or self.paper_trading
    
    def is_backtest_mode(self) -> bool:
        """Check if running in backtest mode."""
        return self.mode == "backtest"


def load_config(config_path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None) -> Config:
    """
    Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to YAML configuration file
        overrides: Dictionary of override values
        
    Returns:
        Loaded and validated configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML configuration
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    # Apply overrides
    if overrides:
        config_data = _apply_overrides(config_data, overrides)
    
    # Create and validate configuration
    config = Config(**config_data)
    
    # Load API credentials from environment
    _load_exchange_credentials(config)
    
    return config


def _apply_overrides(config_data: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply nested overrides to configuration data."""
    for key, value in overrides.items():
        if "." in key:
            # Handle nested keys like "risk.max_position"
            keys = key.split(".")
            current = config_data
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            config_data[key] = value
    
    return config_data


def _load_exchange_credentials(config: Config) -> None:
    """Load exchange credentials from environment variables."""
    for exchange_name in ["binance", "kraken", "coinbase"]:
        exchange_config = config.get_exchange_config(exchange_name)
        
        # Load credentials from environment
        api_key_env = f"{exchange_name.upper()}_API_KEY"
        api_secret_env = f"{exchange_name.upper()}_API_SECRET"
        sandbox_env = f"{exchange_name.upper()}_SANDBOX"
        
        if api_key_env in os.environ:
            exchange_config.api_key = os.environ[api_key_env]
        if api_secret_env in os.environ:
            exchange_config.api_secret = os.environ[api_secret_env]
        if sandbox_env in os.environ:
            exchange_config.sandbox = os.environ[sandbox_env].lower() == "true"
        
        # Coinbase-specific passphrase
        if exchange_name == "coinbase":
            passphrase_env = "COINBASE_PASSPHRASE"
            if passphrase_env in os.environ:
                exchange_config.passphrase = os.environ[passphrase_env]
