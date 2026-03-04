import structlog
from openai import AsyncOpenAI

from .base import EmbeddingProvider

logger = structlog.get_logger()

MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIProvider(EmbeddingProvider):
    """Embedding provider using OpenAI's API."""

    def __init__(self, model: str, api_key: str) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key)
        self._dimensions = MODEL_DIMENSIONS.get(model, 1536)
        logger.info("OpenAI provider initialized", model=model, dimensions=self._dimensions)

    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            input=text,
            model=self._model,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        return [item.embedding for item in response.data]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "openai"
