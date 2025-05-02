import time
from typing import Callable
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Define metrics
REQUEST_COUNT = Counter(
    "api_request_count",
    "Total count of requests by endpoint",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "Request latency in seconds", ["method", "endpoint"]
)

EVENTS_PROCESSED = Counter(
    "events_processed_total", "Total count of processed events", ["status"]
)

ACTIVE_CONNECTIONS = Gauge(
    "active_connections", "Number of active connections to RabbitMQ"
)

SERVICE_INFO = Info("api_gateway_info", "API Gateway service information")

# Set service info
SERVICE_INFO.info({"version": "1.0.0", "service": "api_gateway"})

SQLI_ATTEMPTS = Counter(
    "sqli_attempts_total", "Total SQLi attempts blocked", ["source_ip", "pattern"]
)

XSS_ATTEMPTS = Counter(
    "xss_attempts_total", "Total XSS attempts blocked", ["source_ip", "pattern"]
)

TRAVERSAL_ATTEMPTS = Counter(
    "traversal_attempts_total",
    "Total Path Traversal attempts blocked",
    ["source_ip", "pattern"],
)

DDOS_ATTEMPTS = Counter(
    "ddos_attempts_total", "Total DDoS attempts blocked", ["source_ip", "attack_type"]
)

BLOCKED_REQUESTS = Counter(
    "blocked_requests_total", "Total blocked requests", ["block_type"]
)

BLOCKED_IPS = Gauge("blocked_ips_current", "Current number of blocked IPs", ["reason"])

REQUEST_RATE = Histogram(
    "request_rate_per_ip",
    "Request rate per IP address",
    ["source_ip"],
    buckets=[1, 5, 10, 20, 50, 100],
)


async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    """
    Middleware to capture request metrics.

    Args:
        request: The incoming request
        call_next: The next middleware/endpoint to call

    Returns:
        The response from the endpoint
    """
    method = request.method
    endpoint = request.url.path

    # Exclude metrics endpoint to avoid recursion
    if endpoint == "/metrics":
        return await call_next(request)

    start_time = time.time()

    response = await call_next(request)

    # Record metrics
    status_code = response.status_code
    REQUEST_COUNT.labels(
        method=method, endpoint=endpoint, status_code=status_code
    ).inc()

    # Record latency
    latency = time.time() - start_time
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)

    return response


def setup_metrics(app: FastAPI) -> None:
    """
    Set up the metrics endpoint and middleware.

    Args:
        app: The FastAPI application
    """
    # Add middleware for capturing metrics
    app.middleware("http")(metrics_middleware)

    # Add metrics endpoint
    @app.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def increment_events_processed(status: str = "success") -> None:
    """
    Increment the counter for processed events.

    Args:
        status: The status of the event processing (success, error)
    """
    EVENTS_PROCESSED.labels(status=status).inc()


def set_active_connections(count: int) -> None:
    """
    Set the gauge for active connections.

    Args:
        count: The number of active connections
    """
    ACTIVE_CONNECTIONS.set(count)
