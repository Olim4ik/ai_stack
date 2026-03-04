import structlog

from .base import EmbeddingProvider

logger = structlog.get_logger()


class SentenceTransformersProvider(EmbeddingProvider):
    """Embedding provider using local Sentence Transformers models."""

    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model
        logger.info("Loading Sentence Transformers model", model=model)
        self._model = SentenceTransformer(model)
        self._dimensions = self._model.get_sentence_embedding_dimension()
        logger.info("Model loaded", model=model, dimensions=self._dimensions)

    async def embed(self, text: str) -> list[float]:
        embedding = self._model.encode(text)
        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts)
        return [e.tolist() for e in embeddings]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return "sentence_transformers"
