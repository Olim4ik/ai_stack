"""MCP client wrapper for connecting to the MCP Server."""

import json

import structlog
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = structlog.get_logger()


class MCPClientWrapper:
    """Wraps the MCP client to provide a simple tool-calling interface."""

    def __init__(self, server_command: str, server_cwd: str) -> None:
        self._server_params = StdioServerParameters(
            command=server_command.split()[0],
            args=server_command.split()[1:],
            cwd=server_cwd,
        )
        self._session: ClientSession | None = None
        self._tools: list[dict] = []

    async def connect(self) -> None:
        """Connect to the MCP server and discover available tools."""
        # Note: In production, this would maintain a persistent connection.
        # For now, we reconnect per call for simplicity.
        logger.info("MCP client configured", command=self._server_params.command)

    async def list_tools(self) -> list[dict]:
        """List available tools from the MCP server."""
        async with stdio_client(self._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                self._tools = [
                    {
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.inputSchema,
                    }
                    for t in tools_result.tools
                ]
                return self._tools

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Call a tool on the MCP server and return the result."""
        logger.info("Calling MCP tool", tool=name, arguments=arguments)

        async with stdio_client(self._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)

                # Parse the text content
                if result.content:
                    text = result.content[0].text
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return {"result": text}
                return {"result": None}

    def get_tools_for_llm(self) -> list[dict]:
        """Return tools formatted for Claude's tool_use API."""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in self._tools
        ]
