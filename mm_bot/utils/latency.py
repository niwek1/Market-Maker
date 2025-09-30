"""
Latency measurement and simulation utilities.
"""

import asyncio
import time
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from statistics import mean, median, stdev

import numpy as np


@dataclass
class LatencyStats:
    """Latency statistics container."""
    mean: float
    median: float
    p95: float
    p99: float
    min: float
    max: float
    std: float
    count: int


class LatencyTracker:
    """Track and analyze latency measurements."""
    
    def __init__(self, max_samples: int = 10000):
        """
        Initialize latency tracker.
        
        Args:
            max_samples: Maximum number of samples to keep
        """
        self.max_samples = max_samples
        self.samples: deque = deque(maxlen=max_samples)
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation_id: str) -> None:
        """
        Start timing an operation.
        
        Args:
            operation_id: Unique identifier for the operation
        """
        self.start_times[operation_id] = time.perf_counter()
    
    def end_timer(self, operation_id: str) -> Optional[float]:
        """
        End timing an operation and record latency.
        
        Args:
            operation_id: Unique identifier for the operation
            
        Returns:
            Latency in seconds, None if start time not found
        """
        if operation_id not in self.start_times:
            return None
        
        end_time = time.perf_counter()
        latency = end_time - self.start_times[operation_id]
        
        self.add_sample(latency)
        del self.start_times[operation_id]
        
        return latency
    
    def add_sample(self, latency_seconds: float) -> None:
        """
        Add a latency sample.
        
        Args:
            latency_seconds: Latency in seconds
        """
        self.samples.append(latency_seconds)
    
    def get_stats(self) -> Optional[LatencyStats]:
        """
        Get latency statistics.
        
        Returns:
            LatencyStats object or None if no samples
        """
        if not self.samples:
            return None
        
        samples_array = np.array(self.samples)
        
        return LatencyStats(
            mean=float(np.mean(samples_array)),
            median=float(np.median(samples_array)),
            p95=float(np.percentile(samples_array, 95)),
            p99=float(np.percentile(samples_array, 99)),
            min=float(np.min(samples_array)),
            max=float(np.max(samples_array)),
            std=float(np.std(samples_array)),
            count=len(self.samples)
        )
    
    def get_recent_stats(self, n_samples: int = 100) -> Optional[LatencyStats]:
        """
        Get statistics for recent samples.
        
        Args:
            n_samples: Number of recent samples to analyze
            
        Returns:
            LatencyStats for recent samples
        """
        if not self.samples:
            return None
        
        recent_samples = list(self.samples)[-n_samples:]
        samples_array = np.array(recent_samples)
        
        return LatencyStats(
            mean=float(np.mean(samples_array)),
            median=float(np.median(samples_array)),
            p95=float(np.percentile(samples_array, 95)),
            p99=float(np.percentile(samples_array, 99)),
            min=float(np.min(samples_array)),
            max=float(np.max(samples_array)),
            std=float(np.std(samples_array)),
            count=len(recent_samples)
        )
    
    def clear(self) -> None:
        """Clear all samples and timers."""
        self.samples.clear()
        self.start_times.clear()


class LatencySimulator:
    """Simple latency simulator for backtesting."""
    
    def __init__(self, mean_latency_ms: float = 5.0, std_latency_ms: float = 2.0):
        """Initialize with mean and standard deviation in milliseconds."""
        self.mean_latency_ms = mean_latency_ms
        self.std_latency_ms = std_latency_ms
    
    def get_order_latency(self) -> float:
        """Get simulated order latency in seconds."""
        import random
        latency_ms = max(0.1, random.gauss(self.mean_latency_ms, self.std_latency_ms))
        return latency_ms / 1000.0  # Convert to seconds
    
    def get_market_data_latency(self) -> float:
        """Get simulated market data latency in seconds."""
        import random
        latency_ms = max(0.1, random.gauss(self.mean_latency_ms * 0.5, self.std_latency_ms * 0.5))
        return latency_ms / 1000.0


class NetworkLatencySimulator:
    """Simulate network latency for backtesting."""
    
    def __init__(
        self,
        base_latency_ms: float = 10.0,
        jitter_ms: float = 5.0,
        spike_probability: float = 0.01,
        spike_latency_ms: float = 100.0
    ):
        """
        Initialize network latency simulator.
        
        Args:
            base_latency_ms: Base latency in milliseconds
            jitter_ms: Random jitter in milliseconds
            spike_probability: Probability of latency spike
            spike_latency_ms: Latency during spike in milliseconds
        """
        self.base_latency_ms = base_latency_ms
        self.jitter_ms = jitter_ms
        self.spike_probability = spike_probability
        self.spike_latency_ms = spike_latency_ms
        self.rng = np.random.RandomState()
    
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible results."""
        self.rng = np.random.RandomState(seed)
    
    def sample_latency(self) -> float:
        """
        Sample a latency value.
        
        Returns:
            Latency in seconds
        """
        # Check for spike
        if self.rng.random() < self.spike_probability:
            latency_ms = self.spike_latency_ms
        else:
            # Normal latency with jitter
            jitter = self.rng.normal(0, self.jitter_ms / 3)  # 3-sigma = jitter_ms
            latency_ms = max(0, self.base_latency_ms + jitter)
        
        return latency_ms / 1000.0  # Convert to seconds
    
    async def simulate_delay(self) -> float:
        """
        Simulate network delay asynchronously.
        
        Returns:
            Actual delay applied in seconds
        """
        delay = self.sample_latency()
        await asyncio.sleep(delay)
        return delay


class ExchangeLatencyModel:
    """Model latency characteristics of different exchanges."""
    
    # Typical latency characteristics by exchange (in milliseconds)
    EXCHANGE_PROFILES = {
        "binance": {
            "base_latency": 15,
            "jitter": 8,
            "spike_prob": 0.005,
            "spike_latency": 150
        },
        "kraken": {
            "base_latency": 25,
            "jitter": 12,
            "spike_prob": 0.01,
            "spike_latency": 200
        },
        "coinbase": {
            "base_latency": 30,
            "jitter": 15,
            "spike_prob": 0.008,
            "spike_latency": 180
        },
        "ftx": {
            "base_latency": 20,
            "jitter": 10,
            "spike_prob": 0.006,
            "spike_latency": 120
        },
        "okx": {
            "base_latency": 18,
            "jitter": 9,
            "spike_prob": 0.007,
            "spike_latency": 140
        }
    }
    
    def __init__(self, exchange: str, region: str = "us"):
        """
        Initialize exchange latency model.
        
        Args:
            exchange: Exchange name
            region: Geographic region (us, eu, asia)
        """
        self.exchange = exchange.lower()
        self.region = region.lower()
        
        # Get base profile
        profile = self.EXCHANGE_PROFILES.get(
            self.exchange,
            self.EXCHANGE_PROFILES["binance"]  # Default
        )
        
        # Adjust for region
        region_multiplier = {
            "us": 1.0,
            "eu": 1.2,
            "asia": 1.5,
        }.get(self.region, 1.0)
        
        self.simulator = NetworkLatencySimulator(
            base_latency_ms=profile["base_latency"] * region_multiplier,
            jitter_ms=profile["jitter"] * region_multiplier,
            spike_probability=profile["spike_prob"],
            spike_latency_ms=profile["spike_latency"] * region_multiplier
        )
    
    def set_seed(self, seed: int) -> None:
        """Set random seed."""
        self.simulator.set_seed(seed)
    
    async def order_latency(self) -> float:
        """Get latency for order operations."""
        return await self.simulator.simulate_delay()
    
    async def market_data_latency(self) -> float:
        """Get latency for market data."""
        # Market data typically has lower latency
        base_delay = await self.simulator.simulate_delay()
        return base_delay * 0.7  # 30% faster than order operations


class LatencyBudgetManager:
    """Manage latency budgets for trading operations."""
    
    def __init__(self):
        self.budgets: Dict[str, float] = {}
        self.trackers: Dict[str, LatencyTracker] = {}
    
    def set_budget(self, operation: str, budget_ms: float) -> None:
        """
        Set latency budget for an operation.
        
        Args:
            operation: Operation name
            budget_ms: Budget in milliseconds
        """
        self.budgets[operation] = budget_ms / 1000.0  # Convert to seconds
        if operation not in self.trackers:
            self.trackers[operation] = LatencyTracker()
    
    def start_operation(self, operation: str, operation_id: str) -> None:
        """Start tracking an operation."""
        if operation in self.trackers:
            self.trackers[operation].start_timer(operation_id)
    
    def end_operation(self, operation: str, operation_id: str) -> Tuple[bool, Optional[float]]:
        """
        End tracking an operation and check budget.
        
        Args:
            operation: Operation name
            operation_id: Operation identifier
            
        Returns:
            Tuple of (within_budget, latency_seconds)
        """
        if operation not in self.trackers:
            return True, None
        
        latency = self.trackers[operation].end_timer(operation_id)
        if latency is None:
            return True, None
        
        budget = self.budgets.get(operation)
        if budget is None:
            return True, latency
        
        within_budget = latency <= budget
        return within_budget, latency
    
    def get_budget_utilization(self, operation: str) -> Optional[float]:
        """
        Get average budget utilization for an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Utilization ratio (0-1), None if no data
        """
        if operation not in self.trackers or operation not in self.budgets:
            return None
        
        stats = self.trackers[operation].get_recent_stats(100)
        if stats is None:
            return None
        
        budget = self.budgets[operation]
        return stats.mean / budget
    
    def get_sla_compliance(self, operation: str) -> Optional[float]:
        """
        Get SLA compliance rate for an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Compliance rate (0-1), None if no data
        """
        if operation not in self.trackers or operation not in self.budgets:
            return None
        
        tracker = self.trackers[operation]
        budget = self.budgets[operation]
        
        if not tracker.samples:
            return None
        
        within_budget = sum(1 for sample in tracker.samples if sample <= budget)
        return within_budget / len(tracker.samples)


def measure_async_latency(func):
    """
    Decorator to measure async function latency.
    
    Usage:
        @measure_async_latency
        async def my_function():
            pass
    """
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            latency = end_time - start_time
            # Could log or store latency here
            if hasattr(func, '_latency_tracker'):
                func._latency_tracker.add_sample(latency)
    
    return wrapper


def add_latency_tracking(func, tracker: LatencyTracker):
    """Add latency tracking to a function."""
    func._latency_tracker = tracker
    return measure_async_latency(func)
