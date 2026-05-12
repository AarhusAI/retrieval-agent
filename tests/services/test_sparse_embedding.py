"""Sparse query embedder: provider switching and fastembed integration."""

from unittest.mock import MagicMock, patch

import numpy as np
from qdrant_client.http.models import SparseVector

from app.services import sparse_embedding


async def test_provider_none_returns_all_none():
    """When SPARSE_QUERY_PROVIDER=none, embed_queries returns one None per query."""
    sparse_embedding.close()

    with patch("app.services.sparse_embedding.settings") as mock_settings:
        mock_settings.sparse_query_provider = "none"
        result = await sparse_embedding.embed_queries(["query 1", "query 2"])
    assert result == [None, None]


async def test_fastembed_returns_sparse_vectors():
    """fastembed provider returns SparseVector objects."""
    sparse_embedding.close()

    fake_a = MagicMock()
    fake_a.indices = np.array([1, 5, 7])
    fake_a.values = np.array([0.5, 0.3, 0.2])
    fake_b = MagicMock()
    fake_b.indices = np.array([2, 8])
    fake_b.values = np.array([0.7, 0.1])

    fake_model = MagicMock()
    fake_model.embed.return_value = iter([fake_a, fake_b])

    with (
        patch("app.services.sparse_embedding.settings") as mock_settings,
        patch.object(sparse_embedding, "_model", fake_model),
    ):
        mock_settings.sparse_query_provider = "fastembed"
        mock_settings.sparse_query_model = "Qdrant/bm42-all-minilm-l6-v2-attentions"
        result = await sparse_embedding.embed_queries(["query 1", "query 2"])

    assert len(result) == 2
    assert isinstance(result[0], SparseVector)
    assert result[0].indices == [1, 5, 7]
    assert result[0].values == [0.5, 0.3, 0.2]
    assert isinstance(result[1], SparseVector)
    assert result[1].indices == [2, 8]
    assert result[1].values == [0.7, 0.1]


async def test_preload_noop_when_hybrid_disabled():
    """preload() is a no-op when ENABLE_HYBRID_SEARCH=false."""
    sparse_embedding.close()

    with patch("app.services.sparse_embedding.settings") as mock_settings:
        mock_settings.enable_hybrid_search = False
        mock_settings.sparse_query_provider = "fastembed"
        sparse_embedding.preload()
    assert sparse_embedding._model is None


async def test_preload_noop_when_provider_none():
    """preload() is a no-op when SPARSE_QUERY_PROVIDER=none, even if hybrid is on."""
    sparse_embedding.close()

    with patch("app.services.sparse_embedding.settings") as mock_settings:
        mock_settings.enable_hybrid_search = True
        mock_settings.sparse_query_provider = "none"
        sparse_embedding.preload()
    assert sparse_embedding._model is None
