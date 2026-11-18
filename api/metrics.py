# api/metrics.py — métriques Prometheus
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

HTTP_REQUESTS = Counter("http_requests_total", "Requêtes HTTP", ["method", "path", "status"])
HTTP_LATENCY = Histogram("http_request_duration_seconds", "Latence HTTP", ["path"])
CACHE_EVENTS = Counter("cache_events_total", "Hits/miss du cache", ["result"])
INFER_IN_FLIGHT = Gauge("inference_in_flight", "Inférences en cours")
INFER_WAITING = Gauge("inference_waiting", "Requêtes en attente d'un slot d'inférence")

__all__ = [
    "CONTENT_TYPE_LATEST", "generate_latest", "HTTP_REQUESTS", "HTTP_LATENCY",
    "CACHE_EVENTS", "INFER_IN_FLIGHT", "INFER_WAITING",
]
