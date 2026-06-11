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

import json
import logging
import re

import httpx

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


# --- LLM request payload logging --------------------------------------------
#
# An httpx request event hook that logs the exact JSON body sent to the
# OpenAI-compatible LLM endpoint (model, messages, tools, temperature) at DEBUG.
# This is the readable replacement for openai._base_client's "Request options"
# dump: same payload, one clean line from our own logger, with no httpcore wire
# interleave. The third-party loggers are pinned to an INFO floor in
# ``logging_config`` so that raw firehose stays quiet — see the floor there.
#
# ``app.llm`` is a child of the ``app`` namespace, so it follows the DEBUG bump
# (``DEBUG=true`` or ``LOG_LEVEL=DEBUG``) and is unaffected by that floor.

llm_log = logging.getLogger("app.llm")

# Caps any single string value (the static system prompt, the tools schema, an
# accumulated tool-return blob) so one giant field can't bloat the dump.
_LLM_PAYLOAD_MAX_FIELD_CHARS = 2000


def _truncate_payload(obj: object) -> object:
    """Recursively truncate long string values in a JSON-ish structure."""
    if isinstance(obj, str):
        if len(obj) > _LLM_PAYLOAD_MAX_FIELD_CHARS:
            dropped = len(obj) - _LLM_PAYLOAD_MAX_FIELD_CHARS
            return obj[:_LLM_PAYLOAD_MAX_FIELD_CHARS] + f"… (+{dropped} chars)"
        return obj
    if isinstance(obj, dict):
        return {k: _truncate_payload(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_payload(v) for v in obj]
    return obj


async def log_llm_request(request: httpx.Request) -> None:
    """httpx request hook: log the faithful LLM request payload at DEBUG.

    Attached to the agent's and the query-generation client's httpx transport.
    ``json.dumps`` escapes embedded newlines, so caller content in the payload
    can't forge log lines (same guarantee ``sanitize_for_log`` gives elsewhere).
    Must never raise — logging cannot be allowed to break the actual LLM call.
    """
    try:
        if not llm_log.isEnabledFor(logging.DEBUG):
            return
        body = json.loads(request.content)
        # Only chat-completions-style calls carry ``messages``; skip anything else.
        if not isinstance(body, dict) or "messages" not in body:
            return
        llm_log.debug(
            "LLM request payload (%s):\n%s",
            request.url,
            json.dumps(_truncate_payload(body), indent=2, ensure_ascii=False, default=str),
        )
    except Exception:  # logging must never break the actual LLM call
        pass
