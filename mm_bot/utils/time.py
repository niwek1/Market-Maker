"""
Time utilities for market making operations.
"""

import time
from datetime import datetime, timezone
from typing import Optional, Union

import pandas as pd
import pytz


def utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


def timestamp_ms() -> int:
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


def timestamp_us() -> int:
    """Get current timestamp in microseconds."""
    return int(time.time() * 1_000_000)


def to_timestamp_ms(dt: Union[datetime, pd.Timestamp, str]) -> int:
    """
    Convert datetime to milliseconds timestamp.
    
    Args:
        dt: Datetime object, pandas Timestamp, or ISO string
        
    Returns:
        Timestamp in milliseconds
    """
    if isinstance(dt, str):
        dt = pd.Timestamp(dt)
    elif isinstance(dt, datetime):
        dt = pd.Timestamp(dt)
    
    return int(dt.timestamp() * 1000)


def from_timestamp_ms(ts: int, tz: Optional[str] = None) -> datetime:
    """
    Convert milliseconds timestamp to datetime.
    
    Args:
        ts: Timestamp in milliseconds
        tz: Timezone string (default: UTC)
        
    Returns:
        Datetime object
    """
    dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    
    if tz and tz != "UTC":
        target_tz = pytz.timezone(tz)
        dt = dt.astimezone(target_tz)
    
    return dt


def parse_timestring(time_str: str, tz: str = "UTC") -> datetime:
    """
    Parse time string to datetime object.
    
    Args:
        time_str: Time string in various formats
        tz: Timezone string
        
    Returns:
        Datetime object
    """
    # Try pandas first (handles many formats)
    try:
        dt = pd.Timestamp(time_str)
        if dt.tz is None:
            # Assume UTC if no timezone
            dt = dt.tz_localize("UTC")
        return dt.to_pydatetime()
    except Exception:
        pass
    
    # Fallback to standard datetime parsing
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(time_str, fmt)
            if tz != "UTC":
                target_tz = pytz.timezone(tz)
                dt = target_tz.localize(dt)
            else:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse time string: {time_str}")


def market_hours_filter(
    dt: datetime,
    market: str = "crypto",
    timezone_str: str = "UTC"
) -> bool:
    """
    Check if datetime falls within market hours.
    
    Args:
        dt: Datetime to check
        market: Market type (crypto, forex, stock)
        timezone_str: Market timezone
        
    Returns:
        True if within market hours
    """
    if market == "crypto":
        # Crypto markets are 24/7
        return True
    
    # Convert to market timezone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    if timezone_str != "UTC":
        market_tz = pytz.timezone(timezone_str)
        dt = dt.astimezone(market_tz)
    
    weekday = dt.weekday()
    hour = dt.hour
    
    if market == "forex":
        # Forex: Sunday 5pm EST - Friday 5pm EST
        if weekday == 6:  # Sunday
            return hour >= 22  # 5pm EST = 22 UTC
        elif weekday == 4:  # Friday
            return hour < 22   # Before 5pm EST
        elif weekday in [0, 1, 2, 3]:  # Monday-Thursday
            return True
        else:  # Saturday
            return False
    
    elif market == "stock":
        # US Stock market: Monday-Friday 9:30am-4pm EST
        if weekday >= 5:  # Weekend
            return False
        # 9:30am EST = 14:30 UTC, 4pm EST = 21:00 UTC
        return 14.5 <= hour + dt.minute/60 <= 21
    
    return True


def sleep_until(target_time: Union[datetime, float]) -> None:
    """
    Sleep until target time.
    
    Args:
        target_time: Target datetime or timestamp
    """
    if isinstance(target_time, datetime):
        target_ts = target_time.timestamp()
    else:
        target_ts = target_time
    
    current_ts = time.time()
    sleep_duration = target_ts - current_ts
    
    if sleep_duration > 0:
        time.sleep(sleep_duration)


class Timer:
    """Context manager for measuring execution time."""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.perf_counter()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        
        end_time = self.end_time or time.perf_counter()
        return end_time - self.start_time
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000
    
    @property
    def elapsed_us(self) -> float:
        """Get elapsed time in microseconds."""
        return self.elapsed * 1_000_000


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_second: float):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0.0
    
    async def acquire(self) -> None:
        """Acquire rate limit token (async version)."""
        import asyncio
        
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def acquire_sync(self) -> None:
        """Acquire rate limit token (sync version)."""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()


def align_to_interval(dt: datetime, interval_seconds: int) -> datetime:
    """
    Align datetime to the nearest interval boundary.
    
    Args:
        dt: Datetime to align
        interval_seconds: Interval in seconds
        
    Returns:
        Aligned datetime
    """
    timestamp = int(dt.timestamp())
    aligned_timestamp = (timestamp // interval_seconds) * interval_seconds
    return datetime.fromtimestamp(aligned_timestamp, tz=dt.tzinfo)


def business_days_between(start: datetime, end: datetime) -> int:
    """
    Calculate number of business days between two dates.
    
    Args:
        start: Start datetime
        end: End datetime
        
    Returns:
        Number of business days
    """
    return pd.bdate_range(start=start.date(), end=end.date()).size


def next_business_day(dt: datetime) -> datetime:
    """
    Get next business day.
    
    Args:
        dt: Input datetime
        
    Returns:
        Next business day datetime
    """
    next_day = dt + pd.Timedelta(days=1)
    while next_day.weekday() >= 5:  # Saturday=5, Sunday=6
        next_day += pd.Timedelta(days=1)
    return next_day
