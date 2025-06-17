"""
Memory Management for Live Trading Engine.

This module provides memory management capabilities to prevent memory leaks
during long-running trading sessions. It implements:
- Circular buffers for time-series data
- Automatic cleanup of old data
- Memory usage monitoring
- Garbage collection scheduling
"""

import gc
import logging
import weakref
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Optional

import psutil

logger = logging.getLogger(__name__)


class CircularBuffer:
    """
    A circular buffer implementation for storing time-series data.
    
    Automatically removes old data when the buffer is full.
    """
    
    def __init__(self, maxsize: int):
        """
        Initialize circular buffer.
        
        Args:
            maxsize: Maximum number of items to store
        """
        self.buffer = deque(maxlen=maxsize)
        self.maxsize = maxsize
    
    def append(self, item: Any) -> None:
        """Add item to buffer."""
        self.buffer.append(item)
    
    def get_all(self) -> list:
        """Get all items in buffer."""
        return list(self.buffer)
    
    def clear(self) -> None:
        """Clear all items."""
        self.buffer.clear()
    
    def __len__(self) -> int:
        """Get current size."""
        return len(self.buffer)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"CircularBuffer(size={len(self)}/{self.maxsize})"


class MemoryManager:
    """
    Manages memory usage for the trading system.
    
    Prevents memory leaks and monitors resource usage.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize memory manager.
        
        Args:
            config: Configuration with:
                - max_memory_mb: Maximum memory usage in MB
                - gc_interval: Garbage collection interval in seconds
                - cleanup_interval: Data cleanup interval in seconds
        """
        self.config = config or {}
        self.max_memory_mb = self.config.get("max_memory_mb", 1024)  # 1GB default
        self.gc_interval = self.config.get("gc_interval", 300)  # 5 minutes
        self.cleanup_interval = self.config.get("cleanup_interval", 3600)  # 1 hour
        
        # Track managed objects using weak references
        self._managed_objects = weakref.WeakValueDictionary()
        
        # Memory usage history
        self.memory_history = CircularBuffer(1000)
        
        # Last cleanup times
        self.last_gc_time = datetime.now()
        self.last_cleanup_time = datetime.now()
        
        # Statistics
        self.stats = {
            "gc_runs": 0,
            "cleanups": 0,
            "memory_warnings": 0,
            "peak_memory_mb": 0
        }
    
    def register_object(self, name: str, obj: Any) -> None:
        """
        Register an object for memory management.
        
        Args:
            name: Unique name for the object
            obj: Object to manage
        """
        # Only register objects that can have weak references
        # Skip built-in types like dict, list, str, int, etc.
        try:
            # Test if we can create a weak reference
            weakref.ref(obj)
            self._managed_objects[name] = obj
            logger.debug(f"Registered object for memory management: {name}")
        except TypeError:
            # Can't create weak reference to this type
            logger.debug(f"Skipping registration of {name} - type {type(obj).__name__} doesn't support weak references")
    
    def check_memory_usage(self) -> dict:
        """
        Check current memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = process.memory_percent()
        
        # Update statistics
        self.stats["peak_memory_mb"] = max(self.stats["peak_memory_mb"], memory_mb)
        
        # Store in history
        self.memory_history.append({
            "timestamp": datetime.now(),
            "memory_mb": memory_mb,
            "memory_percent": memory_percent
        })
        
        # Check if we're approaching limits
        if memory_mb > self.max_memory_mb * 0.9:
            self.stats["memory_warnings"] += 1
            logger.warning(f"Memory usage high: {memory_mb:.1f}MB ({memory_percent:.1f}%)")
            
            # Force garbage collection
            self.run_garbage_collection()
        
        return {
            "memory_mb": memory_mb,
            "memory_percent": memory_percent,
            "max_memory_mb": self.max_memory_mb,
            "peak_memory_mb": self.stats["peak_memory_mb"]
        }
    
    def run_garbage_collection(self) -> int:
        """
        Run garbage collection.
        
        Returns:
            Number of objects collected
        """
        # Get initial object count
        before_count = len(gc.get_objects())
        
        # Run garbage collection
        collected = gc.collect()
        
        # Get final object count
        after_count = len(gc.get_objects())
        
        self.stats["gc_runs"] += 1
        self.last_gc_time = datetime.now()
        
        logger.info(f"Garbage collection: {collected} objects collected, "
                   f"total objects: {before_count} → {after_count}")
        
        return collected
    
    def cleanup_old_data(self, retention_hours: int = 24) -> dict:
        """
        Clean up old data from managed objects.
        
        Args:
            retention_hours: Hours of data to retain
        
        Returns:
            Cleanup statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        cleanup_stats = {
            "objects_cleaned": 0,
            "data_removed": 0
        }
        
        # Clean up each managed object
        for name, obj in list(self._managed_objects.items()):
            try:
                # Handle different object types
                if hasattr(obj, "cleanup_old_data"):
                    # Object has its own cleanup method
                    removed = obj.cleanup_old_data(cutoff_time)
                    cleanup_stats["data_removed"] += removed
                    cleanup_stats["objects_cleaned"] += 1
                    
                elif isinstance(obj, dict):
                    # Clean up dictionary entries
                    initial_size = len(obj)
                    
                    # Remove old entries (assuming they have timestamps)
                    for key in list(obj.keys()):
                        if hasattr(obj[key], "timestamp") and obj[key].timestamp < cutoff_time:
                            del obj[key]
                    
                    removed = initial_size - len(obj)
                    if removed > 0:
                        cleanup_stats["data_removed"] += removed
                        cleanup_stats["objects_cleaned"] += 1
                        
                elif isinstance(obj, list):
                    # Clean up list entries
                    initial_size = len(obj)
                    
                    # Remove old entries (assuming they have timestamps)
                    obj[:] = [item for item in obj 
                             if not hasattr(item, "timestamp") or item.timestamp >= cutoff_time]
                    
                    removed = initial_size - len(obj)
                    if removed > 0:
                        cleanup_stats["data_removed"] += removed
                        cleanup_stats["objects_cleaned"] += 1
                        
            except Exception as e:
                logger.error(f"Error cleaning up {name}: {e}")
        
        self.stats["cleanups"] += 1
        self.last_cleanup_time = datetime.now()
        
        logger.info(f"Data cleanup: {cleanup_stats['objects_cleaned']} objects cleaned, "
                   f"{cleanup_stats['data_removed']} items removed")
        
        return cleanup_stats
    
    def should_run_gc(self) -> bool:
        """Check if garbage collection should run."""
        time_since_gc = (datetime.now() - self.last_gc_time).total_seconds()
        return time_since_gc >= self.gc_interval
    
    def should_run_cleanup(self) -> bool:
        """Check if data cleanup should run."""
        time_since_cleanup = (datetime.now() - self.last_cleanup_time).total_seconds()
        return time_since_cleanup >= self.cleanup_interval
    
    def get_memory_report(self) -> dict:
        """
        Get comprehensive memory report.
        
        Returns:
            Dictionary with memory statistics and history
        """
        current_usage = self.check_memory_usage()
        
        # Calculate average memory usage
        history = self.memory_history.get_all()
        if history:
            avg_memory = sum(h["memory_mb"] for h in history) / len(history)
        else:
            avg_memory = current_usage["memory_mb"]
        
        return {
            "current": current_usage,
            "average_mb": avg_memory,
            "statistics": self.stats,
            "last_gc": self.last_gc_time,
            "last_cleanup": self.last_cleanup_time,
            "managed_objects": list(self._managed_objects.keys()),
            "history_size": len(self.memory_history)
        }
    
    def optimize_memory(self) -> dict:
        """
        Run full memory optimization.
        
        Returns:
            Optimization results
        """
        results = {
            "initial_memory": self.check_memory_usage(),
            "gc_collected": 0,
            "data_cleaned": {}
        }
        
        # Run data cleanup
        results["data_cleaned"] = self.cleanup_old_data()
        
        # Run garbage collection
        results["gc_collected"] = self.run_garbage_collection()
        
        # Final memory check
        results["final_memory"] = self.check_memory_usage()
        
        memory_saved = results["initial_memory"]["memory_mb"] - results["final_memory"]["memory_mb"]
        logger.info(f"Memory optimization complete: {memory_saved:.1f}MB saved")
        
        return results