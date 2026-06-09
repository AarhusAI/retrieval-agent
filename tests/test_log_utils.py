"""Tests for the log-injection sanitizer (app/log_utils.py)."""

import pytest

from app.log_utils import sanitize_for_log


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
