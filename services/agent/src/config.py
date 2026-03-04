from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    agent_port: int = 50053
    retrieval_service_host: str = "localhost:50051"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    mcp_server_command: str = "python -m src.main"
    mcp_server_cwd: str = "../mcp_server"

    model_config = {"env_file": ".env"}


settings = Settings()
