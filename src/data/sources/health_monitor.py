"""Health monitoring for data sources - PILLAR 3: OPERATIONAL STABILITY."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SourceHealth:
    """Track health metrics for a data source."""
    name: str
    success_count: int = 0
    failure_count: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    avg_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.failure_count / total
    
    @property
    def is_healthy(self) -> bool:
        """Determine if source is healthy."""
        # Unhealthy if:
        # 1. More than 3 consecutive failures
        if self.consecutive_failures >= 3:
            return False
        
        # 2. Failure rate > 50% in last 10 attempts
        if self.failure_rate > 0.5 and (self.success_count + self.failure_count) >= 10:
            return False
        
        # 3. No success in last 5 minutes
        if self.last_success and datetime.now() - self.last_success > timedelta(minutes=5):
            return False
        
        return True
    
    def update_response_time(self, response_time: float):
        """Update response time metrics."""
        self.response_times.append(response_time)
        # Keep only last 100 response times
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
        self.avg_response_time = sum(self.response_times) / len(self.response_times)


class SourceHealthMonitor:
    """Monitor health of multiple data sources - PILLAR 3: OPERATIONAL STABILITY."""
    
    def __init__(self):
        self.sources: Dict[str, SourceHealth] = {}
    
    def record_success(self, source_name: str, response_time: float = 0.0):
        """Record successful data fetch."""
        if source_name not in self.sources:
            self.sources[source_name] = SourceHealth(name=source_name)
        
        health = self.sources[source_name]
        health.success_count += 1
        health.last_success = datetime.now()
        health.consecutive_failures = 0
        health.update_response_time(response_time)
        
        logger.debug(f"Source {source_name} fetch succeeded (avg response: {health.avg_response_time:.2f}s)")
    
    def record_failure(self, source_name: str, error: Exception):
        """Record failed data fetch."""
        if source_name not in self.sources:
            self.sources[source_name] = SourceHealth(name=source_name)
        
        health = self.sources[source_name]
        health.failure_count += 1
        health.last_failure = datetime.now()
        health.last_error = str(error)
        health.consecutive_failures += 1
        
        logger.warning(
            f"Source {source_name} fetch failed (consecutive: {health.consecutive_failures}): {error}"
        )
    
    def get_healthy_sources(self, sources: List[str]) -> List[str]:
        """Get list of healthy sources in priority order."""
        healthy_sources = []
        
        for source_name in sources:
            if source_name not in self.sources:
                # New source, assume healthy
                healthy_sources.append(source_name)
            elif self.sources[source_name].is_healthy:
                healthy_sources.append(source_name)
        
        # Sort by failure rate (ascending) and response time
        healthy_sources.sort(
            key=lambda s: (
                self.sources.get(s, SourceHealth(s)).failure_rate,
                self.sources.get(s, SourceHealth(s)).avg_response_time
            )
        )
        
        return healthy_sources
    
    def get_status(self) -> Dict[str, Dict]:
        """Get health status of all sources."""
        status = {}
        
        for name, health in self.sources.items():
            status[name] = {
                "healthy": health.is_healthy,
                "success_count": health.success_count,
                "failure_count": health.failure_count,
                "failure_rate": f"{health.failure_rate:.2%}",
                "consecutive_failures": health.consecutive_failures,
                "avg_response_time": f"{health.avg_response_time:.2f}s",
                "last_success": health.last_success.isoformat() if health.last_success else None,
                "last_failure": health.last_failure.isoformat() if health.last_failure else None,
                "last_error": health.last_error
            }
        
        return status