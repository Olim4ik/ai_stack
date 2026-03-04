from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    embedding_port: int = 50052
    embedding_model_provider: str = "openai"
    embedding_model_name: str = "text-embedding-3-small"
    embedding_max_batch_size: int = 64
    openai_api_key: str = ""

    model_config = {"env_file": ".env"}


settings = Settings()
