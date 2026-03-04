from .base import EmbeddingProvider
from .openai_provider import OpenAIProvider
from .st_provider import SentenceTransformersProvider


def create_provider(provider: str, model: str, api_key: str = "") -> EmbeddingProvider:
    """Factory function to create the appropriate embedding provider."""
    match provider:
        case "openai":
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
            return OpenAIProvider(model=model, api_key=api_key)
        case "sentence_transformers":
            return SentenceTransformersProvider(model=model)
        case _:
            raise ValueError(f"Unknown embedding provider: {provider}")
