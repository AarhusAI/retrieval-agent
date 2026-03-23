from unittest.mock import patch

from app.services.qdrant import _get_collection_and_tenant_id


class TestMultitenancyEnabled:
    """Tests with qdrant_multitenancy=True."""

    def setup_method(self):
        self._patcher = patch("app.services.qdrant.settings")
        self.mock_settings = self._patcher.start()
        self.mock_settings.qdrant_multitenancy = True

    def teardown_method(self):
        self._patcher.stop()

    def test_user_memory_collection(self):
        coll, tenant = _get_collection_and_tenant_id("user-memory-abc123", "open-webui")
        assert coll == "open-webui_memories"
        assert tenant == "user-memory-abc123"

    def test_file_collection(self):
        coll, tenant = _get_collection_and_tenant_id("file-xyz", "open-webui")
        assert coll == "open-webui_files"
        assert tenant == "file-xyz"

    def test_web_search_collection(self):
        coll, tenant = _get_collection_and_tenant_id("web-search-query1", "open-webui")
        assert coll == "open-webui_web-search"
        assert tenant == "web-search-query1"

    def test_hex_hash_collection(self):
        hex_name = "a" * 63
        coll, tenant = _get_collection_and_tenant_id(hex_name, "open-webui")
        assert coll == "open-webui_hash-based"
        assert tenant == hex_name

    def test_hex_hash_wrong_length_falls_to_knowledge(self):
        hex_name = "a" * 62
        coll, tenant = _get_collection_and_tenant_id(hex_name, "open-webui")
        assert coll == "open-webui_knowledge"
        assert tenant == hex_name

    def test_hex_hash_non_hex_chars_falls_to_knowledge(self):
        name = "g" * 63  # 'g' is not a hex char
        coll, tenant = _get_collection_and_tenant_id(name, "open-webui")
        assert coll == "open-webui_knowledge"
        assert tenant == name

    def test_knowledge_collection_fallback(self):
        coll, tenant = _get_collection_and_tenant_id("my-knowledge-base", "open-webui")
        assert coll == "open-webui_knowledge"
        assert tenant == "my-knowledge-base"

    def test_custom_prefix(self):
        coll, tenant = _get_collection_and_tenant_id("file-abc", "custom-prefix")
        assert coll == "custom-prefix_files"
        assert tenant == "file-abc"


class TestMultitenancyDisabled:
    """Tests with qdrant_multitenancy=False."""

    def setup_method(self):
        self._patcher = patch("app.services.qdrant.settings")
        self.mock_settings = self._patcher.start()
        self.mock_settings.qdrant_multitenancy = False

    def teardown_method(self):
        self._patcher.stop()

    def test_returns_prefixed_name_no_tenant(self):
        coll, tenant = _get_collection_and_tenant_id("my-collection", "open-webui")
        assert coll == "open-webui_my-collection"
        assert tenant is None

    def test_file_collection_no_multitenancy(self):
        coll, tenant = _get_collection_and_tenant_id("file-abc", "open-webui")
        assert coll == "open-webui_file-abc"
        assert tenant is None
