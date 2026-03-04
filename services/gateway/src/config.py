from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gateway_port: int = 8000
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    embedding_service_host: str = "localhost:50052"
    retrieval_service_host: str = "localhost:50051"
    agent_service_host: str = "localhost:50053"
    chunk_size: int = 512
    chunk_overlap: int = 50

    model_config = {"env_file": ".env"}


settings = Settings()
