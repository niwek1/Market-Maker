"""
Structured logging configuration using structlog and loguru.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from loguru import logger
from structlog.types import FilteringBoundLogger

from mm_bot.config import LoggingConfig


class StructlogHandler(logging.Handler):
    """Handler to bridge Python logging to structlog."""
    
    def __init__(self, logger: FilteringBoundLogger):
        super().__init__()
        self.logger = logger
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record through structlog."""
        level = record.levelname.lower()
        message = record.getMessage()
        
        # Extract extra fields
        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "lineno", "funcName", "created",
                "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "getMessage", "exc_info",
                "exc_text", "stack_info"
            }
        }
        
        # Log through structlog
        log_method = getattr(self.logger, level, self.logger.info)
        log_method(message, **extra)


def setup_logging(config: LoggingConfig) -> FilteringBoundLogger:
    """
    Set up structured logging with both console and file output.
    
    Args:
        config: Logging configuration
        
    Returns:
        Configured structlog logger
    """
    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add appropriate formatter based on config
    if config.format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(colors=True)
        )
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )
    
    # Get the logger
    struct_logger = structlog.get_logger("mm_bot")
    
    # Configure Python logging to use our handler
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(StructlogHandler(struct_logger))
    root_logger.setLevel(getattr(logging, config.level.upper()))
    
    # Configure loguru for file output if enabled
    if config.file_enabled:
        # Remove default handler
        logger.remove()
        
        # Add console handler
        logger.add(
            sys.stderr,
            level=config.level.upper(),
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True,
        )
        
        # Add file handler with rotation
        log_path = Path(config.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            level=config.level.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=config.max_file_size,
            retention=config.backup_count,
            compression="gz",
            serialize=config.format == "json",
        )
    
    return struct_logger


def get_logger(name: str) -> FilteringBoundLogger:
    """Get a named logger instance."""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__name__)
    
    def log_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        fee: float = 0.0,
        order_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log a trade execution."""
        self.logger.info(
            "trade_executed",
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            fee=fee,
            order_id=order_id,
            **kwargs
        )
    
    def log_order(
        self,
        action: str,
        symbol: str,
        side: str,
        size: float,
        price: float,
        order_type: str = "limit",
        order_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log an order action."""
        self.logger.info(
            "order_action",
            action=action,
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            order_type=order_type,
            order_id=order_id,
            **kwargs
        )
    
    def log_risk_event(
        self,
        event_type: str,
        symbol: str,
        current_value: float,
        limit_value: float,
        action: str,
        **kwargs
    ) -> None:
        """Log a risk management event."""
        self.logger.warning(
            "risk_event",
            event_type=event_type,
            symbol=symbol,
            current_value=current_value,
            limit_value=limit_value,
            action=action,
            **kwargs
        )
    
    def log_pnl(
        self,
        symbol: str,
        unrealized_pnl: float,
        realized_pnl: float,
        position: float,
        mark_price: float,
        **kwargs
    ) -> None:
        """Log PnL information."""
        self.logger.info(
            "pnl_update",
            symbol=symbol,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            position=position,
            mark_price=mark_price,
            **kwargs
        )
    
    def log_market_data(
        self,
        symbol: str,
        bid: float,
        ask: float,
        mid: float,
        spread_bps: float,
        **kwargs
    ) -> None:
        """Log market data update."""
        self.logger.debug(
            "market_data",
            symbol=symbol,
            bid=bid,
            ask=ask,
            mid=mid,
            spread_bps=spread_bps,
            **kwargs
        )
    
    def log_performance(
        self,
        metric_name: str,
        value: float,
        period: str = "current",
        **kwargs
    ) -> None:
        """Log performance metrics."""
        self.logger.info(
            "performance_metric",
            metric_name=metric_name,
            value=value,
            period=period,
            **kwargs
        )


def setup_third_party_logging() -> None:
    """Configure logging for third-party libraries."""
    # Reduce noise from common libraries
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    
    # CCXT can be very verbose, filter out routine messages
    ccxt_logger = logging.getLogger("ccxt")
    
    class CCXTFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            message = record.getMessage().lower()
            # Filter out routine CCXT messages
            filtered_phrases = [
                "fetching order book",
                "fetching ticker",
                "fetching balance",
                "rate limit",
                "sleeping",
            ]
            return not any(phrase in message for phrase in filtered_phrases)
    
    ccxt_logger.addFilter(CCXTFilter())
