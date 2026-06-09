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


def test_observability_defaults():
    s = _settings()
    assert s.log_level == "INFO"
    assert s.log_format == "text"
    assert s.metrics_enabled is True
