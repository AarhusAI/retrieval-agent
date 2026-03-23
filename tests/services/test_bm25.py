from app.services.bm25 import _tokenize, reciprocal_rank_fusion


class TestTokenize:
    def test_basic_tokenization(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_single_word(self):
        assert _tokenize("test") == ["test"]

    def test_preserves_punctuation(self):
        tokens = _tokenize("hello, world!")
        assert tokens == ["hello,", "world!"]


class TestReciprocalRankFusion:
    def test_basic_fusion(self):
        vector = [("doc1", 0.9), ("doc2", 0.8)]
        bm25 = [("doc2", 5.0), ("doc3", 3.0)]
        fused = reciprocal_rank_fusion(vector, bm25, bm25_weight=0.5, k_rrf=60)

        texts = [t for t, _ in fused]
        # doc2 appears in both lists, should rank highest
        assert texts[0] == "doc2"
        assert set(texts) == {"doc1", "doc2", "doc3"}

    def test_empty_inputs(self):
        fused = reciprocal_rank_fusion([], [], bm25_weight=0.3)
        assert fused == []

    def test_vector_only(self):
        vector = [("doc1", 0.9), ("doc2", 0.8)]
        fused = reciprocal_rank_fusion(vector, [], bm25_weight=0.3)
        assert len(fused) == 2
        assert fused[0][0] == "doc1"

    def test_bm25_only(self):
        bm25 = [("doc1", 5.0), ("doc2", 3.0)]
        fused = reciprocal_rank_fusion([], bm25, bm25_weight=0.3)
        assert len(fused) == 2
        assert fused[0][0] == "doc1"

    def test_weight_zero_ignores_bm25(self):
        vector = [("doc1", 0.9)]
        bm25 = [("doc2", 5.0)]
        fused = reciprocal_rank_fusion(vector, bm25, bm25_weight=0.0)
        # doc2 gets 0 weight from bm25, so only doc1 has score
        scores = {t: s for t, s in fused}
        assert scores["doc2"] == 0.0

    def test_scores_are_descending(self):
        vector = [("a", 1.0), ("b", 0.5), ("c", 0.1)]
        bm25 = [("c", 10.0), ("b", 5.0), ("a", 1.0)]
        fused = reciprocal_rank_fusion(vector, bm25, bm25_weight=0.5)
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)
