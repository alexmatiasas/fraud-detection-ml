TITLE = "IEEE-CIS Fraud Detection"

SUMMARY = "This is an API to serve a model for fraud detection using the dataset from IEEE CIS"

DESCRIPTION = """
### Description
Links:

- [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
"""
VERSION = "0.1.0"

tags_metadata = [
    {
        "name": "models",
        "description": "The model fetching from here",
    },
    {
        "name": "predict",
        "description": "Manage predictions. So _fancy_ they have their own docs.",
        "externalDocs": {
            "description": "google",
            "url": "https://google.com/",
        },
    },
    {
        "name": "health",
        "description": "Just the health check. If all good, this responds",
        "externalDocs": {
            "description": "google",
            "url": "https://google.com/",
        },
    },
    {
        "name": "ready",
        "description": "An endpoint to verify if this model is ready",
        "externalDocs": {
            "description": "google",
            "url": "https://google.com/",
        },
    },
    {
        "name": "metrics",
        "description": "An endpoint to get metrics from a model",
        "externalDocs": {
            "description": "google",
            "url": "https://google.com/",
        },
    },
]
