"""MCP Server — exposes external tools for the LangGraph Agent."""

import json

import structlog
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .config import settings
from .tools.github import github_get_file, github_get_pr
from .tools.jira import jira_create, jira_search
from .tools.knowledge import knowledge_search
from .tools.pagerduty import pagerduty_ack, pagerduty_incidents

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger()

server = Server("knowledge-base-mcp")

# ── Tool Definitions ────────────────────────────────────────────

TOOLS = [
    Tool(
        name="jira_search",
        description="Search Jira tickets using JQL query syntax",
        inputSchema={
            "type": "object",
            "properties": {
                "jql": {"type": "string", "description": "JQL query (e.g., 'project = PLAT AND status = Open')"},
                "max_results": {"type": "integer", "default": 10},
            },
            "required": ["jql"],
        },
    ),
    Tool(
        name="jira_create",
        description="Create a new Jira ticket. Requires human confirmation.",
        inputSchema={
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Project key (e.g., 'PLAT')"},
                "summary": {"type": "string"},
                "description": {"type": "string", "default": ""},
                "issue_type": {"type": "string", "default": "Task"},
                "priority": {"type": "string", "default": "Medium"},
            },
            "required": ["project", "summary"],
        },
    ),
    Tool(
        name="github_get_pr",
        description="Get details of a GitHub pull request",
        inputSchema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository in 'owner/repo' format"},
                "pr_number": {"type": "integer"},
            },
            "required": ["repo", "pr_number"],
        },
    ),
    Tool(
        name="github_get_file",
        description="Read a file's contents from a GitHub repository",
        inputSchema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository in 'owner/repo' format"},
                "path": {"type": "string", "description": "File path within the repo"},
                "ref": {"type": "string", "default": "main", "description": "Branch or commit SHA"},
            },
            "required": ["repo", "path"],
        },
    ),
    Tool(
        name="pagerduty_incidents",
        description="List current open PagerDuty incidents",
        inputSchema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["triggered", "acknowledged", "resolved"], "default": "triggered"},
                "service_name": {"type": "string", "description": "Filter by service name (optional)", "default": ""},
            },
        },
    ),
    Tool(
        name="pagerduty_ack",
        description="Acknowledge a PagerDuty incident. Requires human confirmation.",
        inputSchema={
            "type": "object",
            "properties": {
                "incident_id": {"type": "string"},
            },
            "required": ["incident_id"],
        },
    ),
    Tool(
        name="knowledge_search",
        description="Search the internal knowledge base for relevant documents",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "team": {"type": "string", "default": "platform", "description": "Team collection to search"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    ),
]

# Tools that require human-in-the-loop confirmation
CONFIRMATION_REQUIRED = {"jira_create", "pagerduty_ack"}


@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    logger.info("Tool called", tool=name, arguments=arguments)

    match name:
        case "jira_search":
            result = await jira_search(
                jql=arguments["jql"],
                max_results=arguments.get("max_results", 10),
                base_url=settings.jira_base_url,
                email=settings.jira_email,
                token=settings.jira_api_token,
            )
        case "jira_create":
            result = await jira_create(
                project=arguments["project"],
                summary=arguments["summary"],
                description=arguments.get("description", ""),
                issue_type=arguments.get("issue_type", "Task"),
                priority=arguments.get("priority", "Medium"),
                base_url=settings.jira_base_url,
                email=settings.jira_email,
                token=settings.jira_api_token,
            )
        case "github_get_pr":
            result = await github_get_pr(
                repo=arguments["repo"],
                pr_number=arguments["pr_number"],
                token=settings.github_token,
            )
        case "github_get_file":
            result = await github_get_file(
                repo=arguments["repo"],
                path=arguments["path"],
                ref=arguments.get("ref", "main"),
                token=settings.github_token,
            )
        case "pagerduty_incidents":
            result = await pagerduty_incidents(
                status=arguments.get("status", "triggered"),
                service_name=arguments.get("service_name", ""),
                api_key=settings.pagerduty_api_key,
            )
        case "pagerduty_ack":
            result = await pagerduty_ack(
                incident_id=arguments["incident_id"],
                api_key=settings.pagerduty_api_key,
            )
        case "knowledge_search":
            result = await knowledge_search(
                query=arguments["query"],
                team=arguments.get("team", "platform"),
                top_k=arguments.get("top_k", 5),
                retrieval_host=settings.retrieval_service_host,
            )
        case _:
            result = json.dumps({"error": True, "code": "UNKNOWN_TOOL", "message": f"Tool '{name}' not found"})

    # Annotate result with confirmation metadata
    result_data = json.loads(result)
    if name in CONFIRMATION_REQUIRED and not result_data.get("error"):
        result_data["_requires_confirmation"] = True
    result = json.dumps(result_data)

    logger.info("Tool result", tool=name, has_error=result_data.get("error", False))
    return [TextContent(type="text", text=result)]


async def main():
    logger.info("Starting MCP Server", transport=settings.mcp_transport)

    if settings.mcp_transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route
        import uvicorn

        sse = SseServerTransport("/messages")

        async def handle_sse(request):
            async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                await server.run(streams[0], streams[1], server.create_initialization_options())

        app = Starlette(routes=[
            Route("/sse", endpoint=handle_sse),
            Route("/messages", endpoint=sse.handle_post_message, methods=["POST"]),
        ])
        uvicorn.run(app, host="0.0.0.0", port=settings.mcp_sse_port)
    else:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
