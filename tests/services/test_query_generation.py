from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from app.models import ChatMessage
from app.services.query_generation import (
    DEFAULT_RETRIEVAL_QUERY_GENERATION_PROMPT_TEMPLATE,
    generate_queries_from_messages,
    render_template,
)


class TestRenderTemplate:
    def test_renders_chat_history_and_date(self):
        messages = [
            ChatMessage(role="user", content="What is RAG?"),
            ChatMessage(role="assistant", content="RAG stands for..."),
        ]
        result = render_template(DEFAULT_RETRIEVAL_QUERY_GENERATION_PROMPT_TEMPLATE, messages)

        assert "user: What is RAG?" in result
        assert "assistant: RAG stands for..." in result
        assert date.today().isoformat() in result

    def test_uses_last_4_messages(self):
        messages = [
            ChatMessage(role="user", content="msg1"),
            ChatMessage(role="assistant", content="msg2"),
            ChatMessage(role="user", content="msg3"),
            ChatMessage(role="assistant", content="msg4"),
            ChatMessage(role="user", content="msg5"),
        ]
        result = render_template(DEFAULT_RETRIEVAL_QUERY_GENERATION_PROMPT_TEMPLATE, messages)

        # msg1 should be excluded (only last 4)
        assert "msg1" not in result
        assert "msg2" in result
        assert "msg5" in result

    def test_empty_messages(self):
        result = render_template(DEFAULT_RETRIEVAL_QUERY_GENERATION_PROMPT_TEMPLATE, [])
        assert date.today().isoformat() in result
        assert "<chat_history>" in result


class TestGenerateQueriesFromMessages:
    async def test_returns_parsed_queries(self):
        mock_choice = MagicMock()
        mock_choice.message.content = '{ "queries": ["budget cuts", "travel policy"] }'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [ChatMessage(role="user", content="tell me about the budget")]

        with patch("app.services.query_generation.AsyncOpenAI", return_value=mock_client):
            queries = await generate_queries_from_messages(messages)

        assert queries == ["budget cuts", "travel policy"]

    async def test_handles_empty_queries(self):
        mock_choice = MagicMock()
        mock_choice.message.content = '{ "queries": [] }'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [ChatMessage(role="user", content="hello!")]

        with patch("app.services.query_generation.AsyncOpenAI", return_value=mock_client):
            queries = await generate_queries_from_messages(messages)

        assert queries == []

    async def test_handles_extra_text_around_json(self):
        mock_choice = MagicMock()
        mock_choice.message.content = 'Here are the queries: { "queries": ["query1"] } Done.'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [ChatMessage(role="user", content="test")]

        with patch("app.services.query_generation.AsyncOpenAI", return_value=mock_client):
            queries = await generate_queries_from_messages(messages)

        assert queries == ["query1"]

    async def test_returns_empty_on_invalid_json(self):
        mock_choice = MagicMock()
        mock_choice.message.content = "I cannot help with that."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [ChatMessage(role="user", content="test")]

        with patch("app.services.query_generation.AsyncOpenAI", return_value=mock_client):
            queries = await generate_queries_from_messages(messages)

        assert queries == []

    async def test_returns_empty_on_llm_exception(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

        messages = [ChatMessage(role="user", content="test")]

        with patch("app.services.query_generation.AsyncOpenAI", return_value=mock_client):
            queries = await generate_queries_from_messages(messages)

        assert queries == []

    async def test_returns_empty_for_empty_messages(self):
        queries = await generate_queries_from_messages([])
        assert queries == []

    async def test_filters_non_string_queries(self):
        mock_choice = MagicMock()
        mock_choice.message.content = '{ "queries": ["valid", 123, "", "also valid"] }'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [ChatMessage(role="user", content="test")]

        with patch("app.services.query_generation.AsyncOpenAI", return_value=mock_client):
            queries = await generate_queries_from_messages(messages)

        assert queries == ["valid", "also valid"]

    async def test_uses_custom_template(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.query_generation.settings.retrieval_query_generation_prompt_template",
            "Custom template: {current_date} {chat_history}",
        )

        mock_choice = MagicMock()
        mock_choice.message.content = '{ "queries": ["custom"] }'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [ChatMessage(role="user", content="test")]

        with patch("app.services.query_generation.AsyncOpenAI", return_value=mock_client):
            queries = await generate_queries_from_messages(messages)

        # Verify the custom template was used in the LLM call
        call_args = mock_client.chat.completions.create.call_args
        prompt_content = call_args.kwargs["messages"][0]["content"]
        assert prompt_content.startswith("Custom template:")
        assert queries == ["custom"]
