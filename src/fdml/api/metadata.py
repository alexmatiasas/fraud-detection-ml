"""OpenAPI metadata for the fraud detection service."""

TITLE = "IEEE-CIS Fraud Detection"
SUMMARY = "REST API serving the IEEE-CIS fraud detection model."
DESCRIPTION = """
# IEEE-CIS Fraud Detection
Serves live fraud predictions from a LightGBM / XGBoost / RandomForest
pipeline trained on the
[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
dataset. Predictions are scored by TransactionID against a labeled sample of
the temporal validation fold, so the full feature pipeline runs server-side.
"""
VERSION = "1.0.0"

#: URL prefix for every API route. Derives from the major version so a bump to
#: ``2.x.y`` moves the API to ``/v2`` — all routers read this single constant.
API_VERSION = VERSION.split(".")[0]
API_PREFIX = f"/v{API_VERSION}"

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
