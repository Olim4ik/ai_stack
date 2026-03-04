"""Unit tests for embedding providers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.embedding.src.providers.base import EmbeddingProvider
from services.embedding.src.providers.factory import create_provider
from services.embedding.src.providers.openai_provider import OpenAIProvider


class TestEmbeddingProviderInterface:
    """Test that the abstract base class enforces the interface."""

    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            EmbeddingProvider()


class TestOpenAIProvider:
    """Test OpenAI embedding provider with mocked API calls."""

    @pytest.fixture
    def mock_openai_response(self):
        """Create a mock OpenAI embedding response."""
        embedding_obj = MagicMock()
        embedding_obj.embedding = [0.1] * 1536
        response = MagicMock()
        response.data = [embedding_obj]
        return response

    @pytest.fixture
    def mock_openai_batch_response(self):
        """Create a mock OpenAI batch embedding response."""
        items = []
        for _ in range(3):
            obj = MagicMock()
            obj.embedding = [0.1] * 1536
            items.append(obj)
        response = MagicMock()
        response.data = items
        return response

    @patch("services.embedding.src.providers.openai_provider.AsyncOpenAI")
    def test_init(self, mock_client_cls):
        provider = OpenAIProvider(model="text-embedding-3-small", api_key="test-key")
        assert provider.model_name == "text-embedding-3-small"
        assert provider.provider_name == "openai"
        assert provider.dimensions == 1536

    @patch("services.embedding.src.providers.openai_provider.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_embed_single(self, mock_client_cls, mock_openai_response):
        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_openai_response)
        mock_client_cls.return_value = mock_client

        provider = OpenAIProvider(model="text-embedding-3-small", api_key="test-key")
        result = await provider.embed("hello world")

        assert len(result) == 1536
        mock_client.embeddings.create.assert_called_once_with(
            input="hello world",
            model="text-embedding-3-small",
        )

    @patch("services.embedding.src.providers.openai_provider.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_embed_batch(self, mock_client_cls, mock_openai_batch_response):
        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_openai_batch_response)
        mock_client_cls.return_value = mock_client

        provider = OpenAIProvider(model="text-embedding-3-small", api_key="test-key")
        texts = ["hello", "world", "test"]
        result = await provider.embed_batch(texts)

        assert len(result) == 3
        assert all(len(v) == 1536 for v in result)


class TestProviderFactory:
    """Test the provider factory function."""

    @patch("services.embedding.src.providers.factory.OpenAIProvider")
    def test_create_openai_provider(self, mock_cls):
        create_provider("openai", "text-embedding-3-small", api_key="test-key")
        mock_cls.assert_called_once_with(model="text-embedding-3-small", api_key="test-key")

    def test_create_openai_without_key_raises(self):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            create_provider("openai", "text-embedding-3-small", api_key="")

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_provider("unknown", "some-model")

    @patch("services.embedding.src.providers.factory.SentenceTransformersProvider")
    def test_create_st_provider(self, mock_cls):
        create_provider("sentence_transformers", "all-MiniLM-L6-v2")
        mock_cls.assert_called_once_with(model="all-MiniLM-L6-v2")
