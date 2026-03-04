from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mcp_transport: str = "stdio"
    mcp_sse_port: int = 8001
    jira_base_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    github_token: str = ""
    pagerduty_api_key: str = ""
    retrieval_service_host: str = "localhost:50051"

    model_config = {"env_file": ".env"}


settings = Settings()
