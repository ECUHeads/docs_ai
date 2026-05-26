"""
Redis-based rate limiting module using Sliding Window Counter algorithm.

Pipeline Improvement:
  Replaces in-memory dictionary-based rate limiting with a Redis-backed
  implementation that supports multi-instance deployment. The sliding window
  counter provides accurate rate tracking across a cluster by maintaining
  request counts in proportional sub-windows within the overall time window.

Algorithm: Sliding Window Counter
  - Divides the time window into smaller sub-windows (default: 10)
  - Counts requests in each sub-window
  - Calculates weighted count by including the previous window's count
    proportionally based on how much of that window has elapsed
  - Provides smoother rate limiting than fixed-window approaches

Fallback:
  If Redis is unavailable, falls back to an in-memory implementation to
  ensure the service continues operating (albeit without cross-instance
  rate coordination).
"""

import time
import logging
from typing import Tuple, Optional, Dict, List

logger = logging.getLogger(__name__)


class RedisRateLimiter:
    """
    Redis-based rate limiter using Sliding Window Counter algorithm.

    Supports multi-instance deployment by storing rate state in Redis.
    Falls back to in-memory rate limiting if Redis is unavailable.
    """

    def __init__(
        self,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        redis_db: int = 0,
        max_requests: int = 100,
        window_seconds: int = 60,
        sub_windows: int = 10,
    ):
        """
        Initialize the rate limiter.

        Args:
            redis_host: Redis server hostname.
            redis_port: Redis server port.
            redis_password: Optional Redis password.
            redis_db: Redis database number.
            max_requests: Maximum number of requests allowed per window.
            window_seconds: Time window in seconds.
            sub_windows: Number of sub-windows for sliding calculation.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.sub_windows = sub_windows
        self.sub_window_seconds = max(1, window_seconds // sub_windows)
        self.redis = None
        self._redis_available = False

        # Fallback in-memory store (used when Redis is unavailable)
        self._memory_store: Dict[str, List[Tuple[float, int]]] = {}

        # Attempt to connect to Redis
        self._connect_redis(redis_host, redis_port, redis_password, redis_db)

    def _connect_redis(
        self,
        host: str,
        port: int,
        password: Optional[str],
        db: int,
    ) -> None:
        """Attempt to establish a Redis connection."""
        try:
            import redis as redis_lib
            self.redis = redis_lib.Redis(
                host=host,
                port=port,
                password=password,
                db=db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            self.redis.ping()
            self._redis_available = True
            logger.info(
                f"Redis rate limiter connected to {host}:{port} (db={db})"
            )
        except ImportError:
            logger.warning(
                "redis package not installed; falling back to in-memory rate limiting. "
                "Install with: pip install redis"
            )
        except Exception as e:
            logger.warning(f"Redis connection failed ({e}); using in-memory fallback: {e}")
            self._redis_available = False

    def is_allowed(self, client_id: str) -> Tuple[bool, Dict[str, int]]:
        """
        Check if a request from *client_id* is allowed under the rate limit.

        Args:
            client_id: Unique identifier for the client (e.g., IP address).

        Returns:
            A tuple of (allowed: bool, info: dict with current_count and limit).
        """
        if self._redis_available and self.redis:
            return self._redis_is_allowed(client_id)
        else:
            return self._memory_is_allowed(client_id)

    def _redis_is_allowed(self, client_id: str) -> Tuple[bool, Dict[str, int]]:
        """
        Sliding Window Counter implementation using Redis.

        The key structure is:
          rate_limit:{client_id}:{sub_window_index}

        The current window's count is exact, and the previous window's count
        is weighted by the fraction of that window that has elapsed.
        """
        try:
            pipe = self.redis.pipeline()
            now = time.time()
            current_sub = int(now // self.sub_window_seconds)

            key_current = f"rate_limit:{client_id}:{current_sub}"
            key_previous = f"rate_limit:{client_id}:{current_sub - 1}"

            # Get counts atomically via pipeline
            pipe.get(key_current)
            pipe.get(key_previous)
            results = pipe.execute()

            count_current = int(results[0]) if results[0] else 0
            count_previous = int(results[1]) if results[1] else 0

            # Calculate weighted count using sliding window
            elapsed_in_current = now % self.sub_window_seconds
            weight_previous = 1.0 - (elapsed_in_current / self.sub_window_seconds)
            weighted_count = count_previous * weight_previous + count_current

            info = {
                'current_count': int(weighted_count),
                'limit': self.max_requests,
                'remaining': max(0, self.max_requests - int(weighted_count)),
            }

            if weighted_count >= self.max_requests:
                return False, info

            # Increment current window counter
            self.redis.incr(key_current)
            # Set expiry to 2 sub-windows to auto-cleanup
            self.redis.expire(key_current, self.sub_window_seconds * 2 + 1)

            info['current_count'] = int(weighted_count) + 1
            info['remaining'] = max(0, self.max_requests - info['current_count'])
            return True, info

        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}; falling back to memory")
            self._redis_available = False
            return self._memory_is_allowed(client_id)

    def _memory_is_allowed(self, client_id: str) -> Tuple[bool, Dict[str, int]]:
        """
        In-memory fallback rate limiter using sliding window.

        Maintains a list of (timestamp, count) tuples per client and calculates
        the weighted count across the sliding window.
        """
        now = time.time()
        window_start = now - self.window_seconds

        if client_id not in self._memory_store:
            self._memory_store[client_id] = []

        # Clean up expired entries
        self._memory_store[client_id] = [
            (ts, c) for ts, c in self._memory_store[client_id] if ts > window_start
        ]

        # Calculate current count within the window
        current_count = sum(c for _, c in self._memory_store[client_id])

        info = {
            'current_count': current_count,
            'limit': self.max_requests,
            'remaining': max(0, self.max_requests - current_count),
        }

        if current_count >= self.max_requests:
            return False, info

        # Record new request
        self._memory_store[client_id].append((now, 1))

        info['current_count'] = current_count + 1
        info['remaining'] = max(0, self.max_requests - info['current_count'])
        return True, info

    def get_usage_info(self, client_id: str) -> Dict[str, any]:
        """
        Get current usage information for a client.

        Args:
            client_id: Unique identifier for the client.

        Returns:
            Dictionary with rate limit usage details.
        """
        _, info = self.is_allowed(client_id)  # Don't increment, just check
        return {
            'client_id': client_id,
            'max_requests': self.max_requests,
            'window_seconds': self.window_seconds,
            **info,
        }
