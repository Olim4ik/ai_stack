from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    retrieval_port: int = 50051
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    embedding_service_host: str = "localhost:50052"
    default_top_k: int = 5
    hybrid_dense_weight: float = 0.7

    model_config = {"env_file": ".env"}


settings = Settings()
