import random
import threading
from dataclasses import dataclass, field


@dataclass
class _VariantStats:
    count: int = 0
    total_latency_ms: float = 0.0
    error_count: int = 0


class ABTestingService:
    """
    Tracks A/B variant assignment and per-variant metrics.
    Thread-safe — uses a lock around stats mutations.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats: dict[str, _VariantStats] = {
            "A": _VariantStats(),
            "B": _VariantStats(),
        }

    def assign_variant(self, split_ratio: float = 0.5) -> str:
        return "A" if random.random() < split_ratio else "B"

    def track_result(self, variant: str, latency_ms: float, error: bool = False):
        with self._lock:
            stats = self._stats.setdefault(variant, _VariantStats())
            stats.count += 1
            stats.total_latency_ms += latency_ms
            if error:
                stats.error_count += 1

    def get_results(self) -> dict:
        with self._lock:
            out = {}
            for variant, stats in self._stats.items():
                avg = stats.total_latency_ms / stats.count if stats.count else 0.0
                err_rate = stats.error_count / stats.count if stats.count else 0.0
                out[variant] = {
                    "count": stats.count,
                    "avg_latency_ms": round(avg, 2),
                    "error_count": stats.error_count,
                    "error_rate": round(err_rate, 4),
                }
            return {"variants": out}

    def reset(self):
        """Mainly for tests."""
        with self._lock:
            self._stats = {"A": _VariantStats(), "B": _VariantStats()}


# module-level singleton
ab_service = ABTestingService()
