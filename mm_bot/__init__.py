"""
Market Making Bot - Professional Avellaneda-Stoikov Implementation

A risk-controlled, exchange-ready market making bot with comprehensive
backtesting and live trading capabilities.
"""

__version__ = "0.1.0"
__author__ = "MM Bot Team"
__email__ = "team@mmbot.dev"

# Core imports for convenience
from mm_bot.config import Config, load_config
from mm_bot.logging import setup_logging
from mm_bot.metrics import setup_metrics

__all__ = [
    "Config",
    "load_config", 
    "setup_logging",
    "setup_metrics",
]
