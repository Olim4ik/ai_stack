from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    agent_port: int = 50053
    retrieval_service_host: str = "localhost:50051"
    openai_api_key: str = ""
    llm_model: str = "gpt-4o"
    mcp_server_command: str = "python -m src.main"
    mcp_server_cwd: str = "../mcp_server"

    model_config = {"env_file": ".env"}


settings = Settings()
