"""
Prometheus-compatible metrics collection and export module.

Pipeline Improvement:
  Provides a /metrics endpoint that exports service health indicators in
  Prometheus text exposition format. Covers key metrics such as:

    - Request counts (success/failure by endpoint)
    - Conversion processing times (histogram)
    - Error rates
    - Service health status

Usage:
  from metrics import get_metrics_collector, metrics_middleware

  # Access collector anywhere in the app
  collector = get_metrics_collector()
  collector.inc_requests_total('convert', '200')
  collector.observe_conversion_time(3.5)

  # Register middleware in aiohttp app
  app = web.Application(middlewares=[metrics_middleware])
"""

import time
import threading
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Thread-safe metrics collector that accumulates counters, gauges, and
    histograms in memory and formats them for Prometheus scraping.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Counters: only increase, never decrease
        self._counters: Dict[str, float] = {
            'requests_total': 0.0,
            'requests_success': 0.0,
            'requests_failed': 0.0,
            'conversions_total': 0.0,
            'conversions_success': 0.0,
            'conversions_failed': 0.0,
            'errors_total': 0.0,
        }

        # Gauges: can go up and down
        self._gauges: Dict[str, float] = {
            'active_requests': 0.0,
            'service_up': 1.0,
        }

        # Histogram buckets for conversion time (seconds)
        self._histogram_buckets: Dict[str, Dict[float, int]] = {
            'conversion_duration_seconds': {
                0.5: 0, 1.0: 0, 2.5: 0, 5.0: 0, 10.0: 0,
                25.0: 0, 50.0: 0, 100.0: 0, 250.0: 0, float('inf'): 0,
            },
        }
        self._histogram_sums: Dict[str, float] = {
            'conversion_duration_seconds': 0.0,
        }
        self._histogram_counts: Dict[str, int] = {
            'conversion_duration_seconds': 0,
        }

        # Per-endpoint request counts
        self._endpoint_requests: Dict[str, Dict[str, int]] = {}

        # Timestamps
        self._start_time = time.time()
        self._last_error_time: Optional[float] = None

    def inc(self, name: str, value: float = 1.0) -> None:
        """Increment a counter by *value*."""
        with self._lock:
            if name in self._counters:
                self._counters[name] += value
            elif name in self._gauges:
                self._gauges[name] += value

    def dec(self, name: str, value: float = 1.0) -> None:
        """Decrement a gauge by *value*."""
        with self._lock:
            if name in self._gauges:
                self._gauges[name] -= value

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge to an absolute value."""
        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        """Record an observation in a histogram."""
        with self._lock:
            if name in self._histogram_buckets:
                for bucket in self._histogram_buckets[name]:
                    if value <= bucket:
                        self._histogram_buckets[name][bucket] += 1
                self._histogram_sums[name] = self._histogram_sums.get(name, 0.0) + value
                self._histogram_counts[name] = self._histogram_counts.get(name, 0) + 1

    # --- Convenience methods ---

    def inc_requests_total(self, endpoint: str, status_code: str) -> None:
        """Track a request by endpoint and HTTP status."""
        self.inc('requests_total')
        key = f"{endpoint}:{status_code}"
        with self._lock:
            self._endpoint_requests.setdefault(key, 0)
            self._endpoint_requests[key] += 1

        if status_code.startswith('2') or status_code.startswith('3'):
            self.inc('requests_success')
        else:
            self.inc('requests_failed')
            self._last_error_time = time.time()

    def observe_conversion_time(self, duration: float) -> None:
        """Record a conversion processing time."""
        self.observe('conversion_duration_seconds', duration)

    def inc_conversions_success(self) -> None:
        """Mark a successful conversion."""
        self.inc('conversions_total')
        self.inc('conversions_success')

    def inc_conversions_failed(self) -> None:
        """Mark a failed conversion."""
        self.inc('conversions_total')
        self.inc('conversions_failed')
        self.inc('errors_total')
        self._last_error_time = time.time()

    # --- Prometheus text exposition format ---

    def generate_prometheus_text(self) -> str:
        """
        Generate metrics in Prometheus text exposition format.

        Returns:
            A string suitable for HTTP response body with Content-Type
            'text/plain; version=0.0.4; charset=utf-8'.
        """
        lines: list[str] = []
        uptime = time.time() - self._start_time
        now_ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

        with self._lock:
            # --- Gauges ---
            lines.append("# HELP service_up Whether the service is up.")
            lines.append("# TYPE service_up gauge")
            lines.append(f"service_up {self._gauges['service_up']}")

            lines.append("# HELP active_requests Number of currently active requests.")
            lines.append("# TYPE active_requests gauge")
            lines.append(f"active_requests {self._gauges['active_requests']}")

            # --- Counters ---
            lines.append("# HELP requests_total Total number of HTTP requests.")
            lines.append("# TYPE requests_total counter")
            lines.append(f"requests_total {self._counters['requests_total']}")

            lines.append("# HELP requests_success_total Total number of successful requests.")
            lines.append("# TYPE requests_success_total counter")
            lines.append(f"requests_success_total {self._counters['requests_success']}")

            lines.append("# HELP requests_failed_total Total number of failed requests.")
            lines.append("# TYPE requests_failed_total counter")
            lines.append(f"requests_failed_total {self._counters['requests_failed']}")

            lines.append("# HELP conversions_total Total number of conversion attempts.")
            lines.append("# TYPE conversions_total counter")
            lines.append(f"conversions_total {self._counters['conversions_total']}")

            lines.append("# HELP conversions_success_total Total number of successful conversions.")
            lines.append("# TYPE conversions_success_total counter")
            lines.append(f"conversions_success_total {self._counters['conversions_success']}")

            lines.append("# HELP conversions_failed_total Total number of failed conversions.")
            lines.append("# TYPE conversions_failed_total counter")
            lines.append(f"conversions_failed_total {self._counters['conversions_failed']}")

            lines.append("# HELP errors_total Total number of errors encountered.")
            lines.append("# TYPE errors_total counter")
            lines.append(f"errors_total {self._counters['errors_total']}")

            # --- Histogram ---
            hist_name = 'conversion_duration_seconds'
            buckets = self._histogram_buckets[hist_name]
            count = self._histogram_counts[hist_name]
            total = self._histogram_sums[hist_name]

            lines.append(f"# HELP {hist_name} Time spent processing PDF conversions (seconds).")
            lines.append(f"# TYPE {hist_name} histogram")

            bucket_parts: list[str] = []
            for boundary in sorted(buckets, key=lambda x: x if x != float('inf') else float('1e18')):
                bucket_parts.append(
                    f'le="{boundary}" {buckets[boundary]}'
                )
            lines.append(
                f"{hist_name}_{{{','.join(bucket_parts)}}}"
            )
            # Re-format properly for Prometheus
            lines.pop()  # Remove the malformed line above
            for boundary in sorted(buckets, key=lambda x: x if x != float('inf') else float('1e18')):
                le_label = "+Inf" if boundary == float('inf') else str(boundary)
                lines.append(
                    f'{hist_name}_bucket{{le="{le_label}"}} {buckets[boundary]}'
                )
            lines.append(f'{hist_name}_count {count}')
            lines.append(f'{hist_name}_sum {total:.6f}')

            # --- Error rate (derived) ---
            total_req = self._counters['requests_total']
            failed_req = self._counters['requests_failed']
            error_rate = failed_req / total_req if total_req > 0 else 0.0

            lines.append("# HELP error_rate Current error rate (failed / total requests).")
            lines.append("# TYPE error_rate gauge")
            lines.append(f"error_rate {error_rate:.6f}")

            # --- Uptime ---
            lines.append("# HELP uptime_seconds Service uptime in seconds.")
            lines.append("# TYPE uptime_seconds gauge")
            lines.append(f"uptime_seconds {uptime:.1f}")

            # --- Timestamp ---
            lines.append("# HELP last_scrape_time Last scrape time in UTC ISO 8601.")
            lines.append("# TYPE last_scrape_time gauge")
            lines.append(f"last_scrape_time{{timestamp=\"{now_ts}\"}} 1")

        return '\n'.join(lines) + '\n'


# --- Singleton ---

_collector: Optional[MetricsCollector] = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global MetricsCollector instance."""
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = MetricsCollector()
    return _collector


# --- aiohttp middleware ---

async def metrics_middleware(
    request,
    handler,
):
    """
    aiohttp middleware that tracks request metrics.

    Increments active_requests on entry and decrements on exit.
    Records endpoint + status code for per-route breakdowns.
    Skips the /metrics endpoint itself to avoid self-counting.
    """
    collector = get_metrics_collector()

    # Skip counting requests to the metrics endpoint itself
    if request.path == '/metrics':
        return await handler(request)

    collector.inc('active_requests')
    start_time = time.time()

    try:
        response = await handler(request)
        duration = time.time() - start_time
        status = str(response.status)
        collector.inc_requests_total(request.path, status)
        return response
    except Exception as e:
        duration = time.time() - start_time
        collector.inc_requests_total(request.path, '500')
        collector.inc('errors_total')
        logger.error(f"Unhandled error in metrics middleware: {e}")
        raise
    finally:
        collector.dec('active_requests')


# --- aiohttp handler for /metrics endpoint ---

async def handle_metrics(request):
    """Handler for the Prometheus /metrics endpoint."""
    from aiohttp import web
    collector = get_metrics_collector()
    text = collector.generate_prometheus_text()
    return web.Response(
        body=text.encode('utf-8'),
        content_type='text/plain; version=0.0.4; charset=utf-8',
    )
