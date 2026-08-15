"""Shared FastAPI dependencies: model loader singleton + API-key guard."""

from __future__ import annotations

import os

from fastapi import Depends, Header, HTTPException

from fdml.api.internal.loader import ModelLoader

_loader: ModelLoader | None = None


def get_model_loader() -> ModelLoader:
    """Singleton accessor — the instance is populated by the app lifespan."""
    global _loader
    if _loader is None:
        _loader = ModelLoader()
    return _loader


def get_loader(loader: ModelLoader = Depends(get_model_loader)) -> ModelLoader:
    """Dependency that fails fast (503) when no model is loaded."""
    if not loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return loader


def verify_api_key(x_api_key: str = Header(...)) -> str:
    """Guard for mutating endpoints — compares against ``FDML_API_KEY``."""
    expected = os.getenv("FDML_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=503, detail="FDML_API_KEY not configured on the server"
        )
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key
