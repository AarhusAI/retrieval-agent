"""Tests for the log-injection sanitizer (app/log_utils.py)."""

import json
import logging

import httpx
import pytest

from app.log_utils import (
    _LLM_PAYLOAD_MAX_FIELD_CHARS,
    log_llm_request,
    sanitize_for_log,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Normal strings pass through untouched.
        ("normal-query", "normal-query"),
        ("file-abc-123", "file-abc-123"),
        # Newlines / CR — the classic log-injection vector.
        ("foo\nfake [ERROR] log line", "foo?fake [ERROR] log line"),
        ("foo\r\nbar", "foo??bar"),
        # ANSI CSI escape — would otherwise let an attacker recolour log
        # output to fool grep / dashboards.
        ("foo\x1b[31mRED", "foo?[31mRED"),
        # NUL byte.
        ("foo\x00bar", "foo?bar"),
        # DEL (0x7F).
        ("foo\x7fbar", "foo?bar"),
        # Bytes above 0x7F (non-ASCII letters) are left alone — log handlers
        # encode them safely and stripping them would mangle Danish queries.
        ("Hvad er en å-rapport?", "Hvad er en å-rapport?"),
        # None becomes empty string.
        (None, ""),
        # Non-strings (e.g. a list of collection names) are coerced.
        (["file-1", "file-2"], "['file-1', 'file-2']"),
        (123, "123"),
    ],
)
def test_sanitize_for_log_strips_controls(raw, expected):
    assert sanitize_for_log(raw) == expected


def test_sanitize_for_log_truncates():
    """Long values get a ``...`` marker so a megabyte payload can't bloat the log."""
    long_value = "A" * 1000
    out = sanitize_for_log(long_value, max_len=50)
    assert out.endswith("...")
    assert len(out) == 53  # 50 + "..."


# --- log_llm_request hook ---------------------------------------------------

_URL = "https://litellm.example/v1/chat/completions"


@pytest.fixture
def _restore_llm_level():
    """log_llm_request mutates nothing, but tests flip the app.llm level —
    snapshot and restore so they don't leak into the rest of the suite."""
    logger = logging.getLogger("app.llm")
    saved = logger.level
    yield logger
    logger.setLevel(saved)


def _request(body: dict | bytes) -> httpx.Request:
    if isinstance(body, bytes):
        return httpx.Request("POST", _URL, content=body)
    return httpx.Request("POST", _URL, json=body)


async def test_log_llm_request_logs_faithful_payload(caplog, _restore_llm_level):
    body = {
        "model": "AarhusAI-default-v2",
        "temperature": 0,
        "messages": [{"role": "user", "content": "hej"}],
        "tools": [{"function": {"name": "retrieve"}}],
    }
    with caplog.at_level(logging.DEBUG, logger="app.llm"):
        await log_llm_request(_request(body))

    records = [r for r in caplog.records if "LLM request payload" in r.getMessage()]
    assert len(records) == 1
    logged = records[0].getMessage()
    assert "AarhusAI-default-v2" in logged
    assert "retrieve" in logged
    # The payload is rendered as JSON, which escapes newlines (no log-forging).
    assert "\\n" in json.dumps("a\nb")  # sanity on the escaping we rely on


async def test_log_llm_request_truncates_long_fields(caplog, _restore_llm_level):
    long_prompt = "x" * (_LLM_PAYLOAD_MAX_FIELD_CHARS + 500)
    body = {"model": "m", "messages": [{"role": "system", "content": long_prompt}]}
    with caplog.at_level(logging.DEBUG, logger="app.llm"):
        await log_llm_request(_request(body))

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "(+500 chars)" in logged
    assert long_prompt not in logged  # the full untruncated value never appears


async def test_log_llm_request_skips_non_chat_body(caplog, _restore_llm_level):
    """A body without ``messages`` (e.g. an embeddings call) is not logged."""
    with caplog.at_level(logging.DEBUG, logger="app.llm"):
        await log_llm_request(_request({"input": "embed me", "model": "e5"}))
    assert not [r for r in caplog.records if "LLM request payload" in r.getMessage()]


async def test_log_llm_request_swallows_malformed_body(caplog, _restore_llm_level):
    """A non-JSON body must not raise and must not log."""
    with caplog.at_level(logging.DEBUG, logger="app.llm"):
        await log_llm_request(_request(b"not json at all"))  # must not raise
    assert not [r for r in caplog.records if "LLM request payload" in r.getMessage()]


async def test_log_llm_request_silent_when_not_debug(caplog, _restore_llm_level):
    """Below DEBUG the hook returns early — no payload work, no record."""
    logging.getLogger("app.llm").setLevel(logging.INFO)
    body = {"model": "m", "messages": [{"role": "user", "content": "hej"}]}
    with caplog.at_level(logging.INFO, logger="app.llm"):
        await log_llm_request(_request(body))
    assert not [r for r in caplog.records if "LLM request payload" in r.getMessage()]
