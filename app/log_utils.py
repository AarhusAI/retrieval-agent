"""Logging helpers — small enough to live at the app root.

``sanitize_for_log`` is the single defense against log-injection via
caller-controlled values (``collection_names`` and, at DEBUG, the raw
query text the caller sends). Newlines and ANSI escapes embedded in those
values would let an attacker forge fake log records that confuse downstream
parsers; truncation prevents a megabyte-sized field from bloating the log.

Applied at log call sites rather than in a logging.Formatter so this
helper has no opinion about which logger / handler is configured and
plays nicely with the JSON formatter.
"""

from __future__ import annotations

import re

# C0 controls (0x00-0x1F) plus DEL (0x7F). Covers newlines, carriage
# returns, NUL, and the ANSI CSI introducer (0x1B). Bytes above 0x7F
# (UTF-8 continuation, printable Unicode) are left alone — log handlers
# encode them safely and stripping them would mangle non-ASCII queries.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def sanitize_for_log(value: object, max_len: int = 200) -> str:
    """Coerce ``value`` to a printable single-line string ≤ ``max_len`` chars.

    - ``None`` → ``""``.
    - Control characters → ``"?"``.
    - Strings longer than ``max_len`` are truncated with a ``"..."`` marker.
    - Non-strings are coerced via ``str()``.
    """
    if value is None:
        return ""
    s = str(value)
    sanitized = _CONTROL_CHARS.sub("?", s)
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len] + "..."
    return sanitized
