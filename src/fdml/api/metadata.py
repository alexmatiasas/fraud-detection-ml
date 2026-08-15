"""OpenAPI metadata for the fraud detection service."""

TITLE = "IEEE-CIS Fraud Detection"
SUMMARY = "REST API serving the IEEE-CIS fraud detection model."
DESCRIPTION = """
# IEEE-CIS Fraud Detection
Serves live fraud predictions from a LightGBM pipeline trained on the
[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
dataset. Predictions are scored by TransactionID against a labeled sample of
the temporal validation fold, so the full feature pipeline runs server-side.
"""
VERSION = "0.1.1"

tags_metadata = [
    {
        "name": "predict",
        "description": "Fraud prediction by TransactionID",
    },
    {
        "name": "model",
        "description": "Model metadata and reload",
    },
    {
        "name": "metrics",
        "description": "Offline evaluation metrics from models/report.json",
    },
    {
        "name": "health",
        "description": "Liveness probe — the API process is up",
    },
    {
        "name": "ready",
        "description": "Readiness probe — the model is loaded",
    },
]
