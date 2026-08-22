from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from fdml.api.dependencies import (
    get_loader,
    get_model_loader,
    get_multi_loader,
    verify_api_key,
)
from fdml.api.internal.loader import ModelLoader


class TestGetModelLoader:
    def test_creates_singleton(self):
        import fdml.api.dependencies as dep_mod

        dep_mod._loader = None
        loader = get_model_loader()
        assert isinstance(loader, ModelLoader)
        assert get_model_loader() is loader

    def test_multi_loader_creates_singleton(self):
        import fdml.api.dependencies as dep_mod

        dep_mod._multi_loader = None
        loader = get_multi_loader()
        assert loader is get_multi_loader()


class TestGetLoader:
    def test_raises_when_not_loaded(self):
        loader = MagicMock(spec=ModelLoader)
        loader.is_loaded = False
        with pytest.raises(HTTPException) as exc_info:
            get_loader(loader=loader)
        assert exc_info.value.status_code == 503

    def test_returns_when_loaded(self):
        loader = MagicMock(spec=ModelLoader)
        loader.is_loaded = True
        assert get_loader(loader=loader) is loader


class TestVerifyApiKey:
    def test_raises_when_not_configured(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(HTTPException) as exc_info:
                verify_api_key(x_api_key="test")
            assert exc_info.value.status_code == 503

    def test_raises_on_wrong_key(self):
        with patch.dict("os.environ", {"FDML_API_KEY": "correct-key"}):
            with pytest.raises(HTTPException) as exc_info:
                verify_api_key(x_api_key="wrong")
            assert exc_info.value.status_code == 401

    def test_returns_key_on_match(self):
        with patch.dict("os.environ", {"FDML_API_KEY": "correct-key"}):
            assert verify_api_key(x_api_key="correct-key") == "correct-key"


class TestLifespan:
    def test_startup_loads_model(self):
        from fdml.api.main import lifespan
        from fdml.api.main import app as _app

        mock_loader = MagicMock()
        mock_loader.is_loaded = True
        mock_multi = MagicMock()
        with (
            patch("fdml.api.main.get_model_loader", return_value=mock_loader),
            patch("fdml.api.main.get_multi_loader", return_value=mock_multi),
        ):
            import asyncio

            ctx = lifespan(_app)
            asyncio.get_event_loop().run_until_complete(ctx.__aenter__())
            mock_loader.load.assert_called_once()
            mock_loader.load_sample.assert_called_once()
            mock_multi.load.assert_called_once()

    def test_startup_handles_load_error(self):
        from fdml.api.main import lifespan
        from fdml.api.main import app as _app
        from fdml.api.internal.loader import ModelLoadError

        mock_loader = MagicMock()
        mock_loader.load.side_effect = ModelLoadError("no model")
        mock_multi = MagicMock()
        with (
            patch("fdml.api.main.get_model_loader", return_value=mock_loader),
            patch("fdml.api.main.get_multi_loader", return_value=mock_multi),
        ):
            import asyncio

            ctx = lifespan(_app)
            asyncio.get_event_loop().run_until_complete(ctx.__aenter__())
            mock_loader.load_sample.assert_called_once()

    def test_startup_handles_multi_error(self):
        from fdml.api.main import lifespan
        from fdml.api.main import app as _app
        from fdml.api.internal.registry import ModelRegistryError

        mock_loader = MagicMock()
        mock_loader.is_loaded = True
        mock_multi = MagicMock()
        mock_multi.load.side_effect = ModelRegistryError("no registry")
        with (
            patch("fdml.api.main.get_model_loader", return_value=mock_loader),
            patch("fdml.api.main.get_multi_loader", return_value=mock_multi),
        ):
            import asyncio

            ctx = lifespan(_app)
            asyncio.get_event_loop().run_until_complete(ctx.__aenter__())
            mock_multi.load.assert_called_once()
