"""
LLM-based query generation from chat messages.

Used by the linear pipeline when messages are received without pre-generated
queries (i.e. Open WebUI bypass mode). Mirrors the logic of Open WebUI's
RETRIEVAL_QUERY_GENERATION_PROMPT_TEMPLATE.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from string import Template

import httpx
from openai import AsyncOpenAI

from app.config import settings
from app.log_utils import log_llm_request, sanitize_for_log
from app.models import ChatMessage

log = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None
# Our own httpx transport so we can attach the request hook that logs the
# faithful LLM payload at DEBUG (app.llm). Held separately because AsyncOpenAI
# does not own a caller-supplied http_client — we close it ourselves.
_http_client: httpx.AsyncClient | None = None


def get_client() -> AsyncOpenAI:
    global _client, _http_client
    if _client is None:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            event_hooks={"request": [log_llm_request]},
        )
        _client = AsyncOpenAI(
            base_url=settings.agent_api_base_url or None,
            api_key=settings.agent_api_key or "unused",
            http_client=_http_client,
        )
    return _client


async def close_client() -> None:
    global _client, _http_client
    if _client is not None:
        await _client.close()
        _client = None
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


DEFAULT_RETRIEVAL_QUERY_GENERATION_PROMPT_TEMPLATE = """\
### Task:
Analyze the chat history and generate 1-3 search queries optimized for \
retrieving relevant documents from a knowledge base using semantic vector search.

### Guidelines:
- Respond **EXCLUSIVELY** with a JSON object.
- Base queries on the **user's questions and information needs only**. \
Use assistant responses solely for context and disambiguation \
(e.g. resolving "that", "it", "the one you mentioned").
- Generate queries as natural-language phrases that capture the semantic \
meaning of the user's information need.
- Reformulate conversational references into standalone, self-contained queries.
- Each query should target a different aspect or angle to maximize retrieval coverage.
- If the user's message clearly needs no document retrieval \
(e.g. greetings), return: { "queries": [] }
- Respond in the same language as the user's messages.
- Today's date is: $current_date

### Output:
{ "queries": ["query1", "query2"] }

### Chat History:
<chat_history>
$chat_history
</chat_history>\
"""


def _get_template(override: str | None = None) -> str:
    """Return the template to use, checking: request override > env var > default."""
    if override and override.strip():
        return override.strip()
    custom = settings.retrieval_query_generation_prompt_template.strip()
    return custom if custom else DEFAULT_RETRIEVAL_QUERY_GENERATION_PROMPT_TEMPLATE


def render_template(template: str, messages: list[ChatMessage]) -> str:
    """Render the query generation template with chat history and date.

    Uses ``string.Template`` (``$name`` placeholders) instead of
    ``str.format`` so that an attacker-controlled template (passed through
    via the request body or env var) cannot reach into Python's attribute
    machinery — ``str.format`` honours ``{x.__class__.__bases__[0].__subclasses__()}``
    style expressions and would otherwise let any caller with the bearer
    walk to ``settings`` and exfiltrate upstream API keys. ``safe_substitute``
    leaves unknown ``$`` placeholders untouched rather than raising.
    """
    # Use last 4 messages (matching Open WebUI's {{MESSAGES:END:4}})
    recent = messages[-4:]
    chat_history = "\n".join(f"{m.role}: {m.content}" for m in recent)
    return Template(template).safe_substitute(
        current_date=date.today().isoformat(),
        chat_history=chat_history,
    )


async def generate_queries_from_messages(
    messages: list[ChatMessage],
    template_override: str | None = None,
) -> list[str]:
    """
    Generate optimized retrieval queries from chat messages via LLM.

    Args:
        messages: Chat messages to generate queries from.
        template_override: Optional template from the request (e.g. passed by Open WebUI).
            Takes precedence over env var and default.

    Returns a list of queries, or an empty list on failure (caller should
    fall back to extracting the last user message).
    """
    if not messages:
        return []

    template = _get_template(template_override)
    prompt = render_template(template, messages)

    try:
        client = get_client()
        response = await client.chat.completions.create(
            model=settings.agent_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content or ""

        if response.usage:
            log.debug(
                "Query generation token usage: model=%s, prompt_tokens=%d, "
                "completion_tokens=%d, total_tokens=%d",
                settings.agent_model,
                response.usage.prompt_tokens or 0,
                response.usage.completion_tokens or 0,
                response.usage.total_tokens or 0,
            )

        # Parse JSON — tolerant of extra text around the JSON object
        bracket_start = content.find("{")
        bracket_end = content.rfind("}") + 1
        if bracket_start == -1 or bracket_end <= 0:
            log.warning("No JSON object in query generation response")
            log.debug("Query generation response payload: %s", sanitize_for_log(content))
            return []

        parsed = json.loads(content[bracket_start:bracket_end])
        queries = parsed.get("queries", [])

        if not isinstance(queries, list):
            log.warning("Unexpected queries type: %s", type(queries))
            return []

        queries = [q for q in queries if isinstance(q, str) and q.strip()]
        log.info("Generated %d queries from messages", len(queries))
        log.debug("Generated queries: %s", sanitize_for_log(queries))
        return queries

    except Exception:
        log.exception("Failed to generate queries from messages")
        return []
