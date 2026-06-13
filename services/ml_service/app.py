import os
import time

from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from ml_service.handler import RecommenderHandler
from ml_service.schemas import ClientFeatures, RecommendationResponse


MODEL_PATH = os.getenv("MODEL_PATH", "models/recommender.pkl")

app = FastAPI(
    title="Bank Product Recommendation Service",
    description="REST API для рекомендации банковских продуктов",
    version="1.0.0",
)

handler = RecommenderHandler(MODEL_PATH)

recommendation_counter = Counter(
    "recommendation_requests_total",
    "Total number of recommendation requests",
)
recommendation_latency = Histogram(
    "recommendation_latency_seconds",
    "Recommendation latency in seconds",
)
average_renta = Gauge(
    "client_renta_avg",
    "Average renta in incoming requests",
)
average_products = Gauge(
    "client_products_avg",
    "Average number of owned products in incoming requests",
)

Instrumentator().instrument(app).expose(app)


@app.get("/")
def root():
    return {"status": "ok", "service": "bank_product_recommendations"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(client: ClientFeatures):
    start_time = time.time()
    payload = client.model_dump()

    recommendations = handler.recommend(payload)
    snapshot = handler.feature_snapshot(payload)

    recommendation_counter.inc()
    recommendation_latency.observe(time.time() - start_time)
    average_renta.set(snapshot["renta"])
    average_products.set(snapshot["n_products"])

    return {
        "ncodpers": client.ncodpers,
        "recommendations": recommendations,
    }
