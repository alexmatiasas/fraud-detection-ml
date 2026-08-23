"""Locust load tests for the fraud detection API.

Run locally:
    uv run locust -f tests/load/locustfile.py --host http://localhost:8000

Run in CI (headless, fixed duration):
    uv run locust -f tests/load/locustfile.py --host $BASE_URL \\
        --headless -u 10 -r 2 --run-time 30s --csv=results
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task


DEMO_TRANSACTION_IDS = [3538759, 3538812, 3538861, 3538920, 3538957]


class FraudAPIUser(HttpUser):
    """Simulates a user interacting with the fraud detection API."""

    wait_time = between(0.5, 2)

    @task(5)
    def predict(self):
        """Score a single transaction (highest-traffic endpoint)."""
        self.client.post(
            "/v1/predict/",
            json={
                "transaction_id": random.choice(DEMO_TRANSACTION_IDS),
            },
            name="/v1/predict/ [single]",
        )

    @task(2)
    def batch_predict(self):
        """Score a small batch of transactions."""
        n = random.randint(2, 5)
        self.client.post(
            "/v1/predict/batch",
            json=[
                {"transaction_id": tid}
                for tid in random.sample(DEMO_TRANSACTION_IDS, n)
            ],
            name="/v1/predict/batch",
        )

    @task(4)
    def list_transactions(self):
        """Fetch the transaction selector page."""
        self.client.get(
            "/v1/transactions/?limit=20",
            name="/v1/transactions/",
        )

    @task(1)
    def list_models(self):
        """Fetch the model leaderboard."""
        self.client.get("/v1/models/", name="/v1/models/")

    @task(1)
    def health(self):
        """Liveness probe."""
        self.client.get("/v1/health/", name="/v1/health/")

    @task(1)
    def ready(self):
        """Readiness probe."""
        self.client.get("/v1/ready/", name="/v1/ready/")
