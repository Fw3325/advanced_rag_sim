"""
Timing utilities for profiling and performance monitoring.
"""
import time
from contextlib import contextmanager
from typing import Dict

class TimingProfiler:
    """Simple profiler for tracking operation times."""
    
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
    
    @contextmanager
    def time(self, operation: str):
        """Context manager to time an operation."""
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            if operation not in self.timings:
                self.timings[operation] = 0.0
                self.counts[operation] = 0
            self.timings[operation] += elapsed
            self.counts[operation] += 1
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics."""
        stats = {}
        for op, total_time in self.timings.items():
            count = self.counts[op]
            stats[op] = {
                "total": total_time,
                "count": count,
                "avg": total_time / count if count > 0 else 0
            }
        return stats
    
    def print_stats(self):
        """Print timing statistics."""
        print("\n" + "="*50)
        print("PERFORMANCE PROFILING")
        print("="*50)
        for op, stats in self.get_stats().items():
            print(f"{op:30s} | Total: {stats['total']:6.2f}s | Avg: {stats['avg']:6.2f}s | Count: {stats['count']}")
        print("="*50 + "\n")
    
    def reset(self):
        """Reset all timings."""
        self.timings.clear()
        self.counts.clear()

# Global profiler instance
_profiler = TimingProfiler()

def get_profiler() -> TimingProfiler:
    """Get global profiler instance."""
    return _profiler

def reset_profiler():
    """Reset global profiler."""
    _profiler.reset()
