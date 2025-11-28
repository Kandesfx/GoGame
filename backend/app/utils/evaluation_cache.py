"""Evaluation cache để tối ưu performance cho premium features."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class EvaluationCache:
    """LRU Cache với TTL cho board evaluations."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0):
        """
        Args:
            max_size: Số lượng entries tối đa trong cache
            ttl_seconds: Time-to-live cho mỗi entry (seconds)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[int, Tuple[float, Any]] = OrderedDict()  # hash -> (timestamp, value)
        self.hits = 0
        self.misses = 0

    def get(self, cache_key: Any) -> Optional[Any]:
        """Lấy evaluation từ cache nếu còn valid.
        
        Args:
            cache_key: Key để lookup (có thể là int hoặc hashable object)
        """
        # Convert key to hash nếu cần
        if not isinstance(cache_key, int):
            cache_key = hash(cache_key)
            
        if cache_key not in self.cache:
            self.misses += 1
            return None

        timestamp, value = self.cache[cache_key]

        # Kiểm tra TTL
        if time.time() - timestamp > self.ttl_seconds:
            # Expired, remove
            del self.cache[cache_key]
            self.misses += 1
            return None

        # Move to end (LRU)
        self.cache.move_to_end(cache_key)
        self.hits += 1
        return value

    def set(self, cache_key: Any, value: Any) -> None:
        """Lưu evaluation vào cache.
        
        Args:
            cache_key: Key để lưu (có thể là int hoặc hashable object)
            value: Giá trị cần cache
        """
        # Convert key to hash nếu cần
        if not isinstance(cache_key, int):
            cache_key = hash(cache_key)
            
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size and cache_key not in self.cache:
            self.cache.popitem(last=False)  # Remove oldest

        self.cache[cache_key] = (time.time(), value)
        self.cache.move_to_end(cache_key)

    def clear(self) -> None:
        """Xóa toàn bộ cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê cache."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 3),
        }

    def cleanup_expired(self) -> int:
        """Xóa các entries đã expired. Trả về số lượng đã xóa."""
        current_time = time.time()
        expired_keys = [
            key for key, (timestamp, _) in self.cache.items() if current_time - timestamp > self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]
        return len(expired_keys)


# Global cache instance
_global_cache: Optional[EvaluationCache] = None


def get_evaluation_cache(max_size: int = 1000, ttl_seconds: float = 3600.0) -> EvaluationCache:
    """Lấy global evaluation cache instance (singleton pattern)."""
    global _global_cache
    if _global_cache is None:
        _global_cache = EvaluationCache(max_size=max_size, ttl_seconds=ttl_seconds)
    return _global_cache


def clear_evaluation_cache() -> None:
    """Xóa global cache (dùng cho testing hoặc reset)."""
    global _global_cache
    if _global_cache:
        _global_cache.clear()

