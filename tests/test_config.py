"""Observability config validators (app/config.py)."""

import pytest
from pydantic import ValidationError

from app.config import Settings


def _settings(**overrides) -> Settings:
    base = {"api_key": "a" * 32}
    base.update(overrides)
    return Settings(_env_file=None, **base)


def test_log_level_normalises_case():
    assert _settings(log_level="debug").log_level == "DEBUG"


def test_log_level_rejects_unknown():
    with pytest.raises(ValidationError):
        _settings(log_level="verbose")


def test_log_format_normalises_case():
    assert _settings(log_format="JSON").log_format == "json"


def test_log_format_rejects_unknown():
    with pytest.raises(ValidationError):
        _settings(log_format="yaml")


def test_log_level_app_normalises_case():
    assert _settings(log_level_app="debug").log_level_app == "DEBUG"


def test_log_level_app_empty_stays_empty():
    assert _settings(log_level_app="").log_level_app == ""


def test_log_level_app_rejects_unknown():
    with pytest.raises(ValidationError):
        _settings(log_level_app="verbose")


def test_embedding_base_url_rejects_empty():
    with pytest.raises(ValidationError):
        _settings(embedding_api_base_url="")


def test_embedding_base_url_rejects_whitespace():
    with pytest.raises(ValidationError):
        _settings(embedding_api_base_url="   ")


def test_observability_defaults():
    s = _settings()
    assert s.log_level == "INFO"
    assert s.log_level_app == ""
    assert s.log_format == "text"
    assert s.metrics_enabled is True
