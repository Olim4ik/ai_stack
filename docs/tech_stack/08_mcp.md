# 08. MCP (Model Context Protocol)

> **Interview Preparation Document for Backend AI Engineers**
> Covers theory, architecture, code examples, and Q&A for MCP -- Anthropic's open
> standard for connecting LLMs with external tools, data sources, and services.

---

## Table of Contents

1. [MCP Overview](#1-mcp-overview)
2. [Core Concepts](#2-core-concepts)
3. [MCP Server Development](#3-mcp-server-development)
4. [MCP Client Development](#4-mcp-client-development)
5. [Resources in Detail](#5-resources-in-detail)
6. [Tools in Detail](#6-tools-in-detail)
7. [Prompts in Detail](#7-prompts-in-detail)
8. [Transport Protocols](#8-transport-protocols)
9. [MCP in Production](#9-mcp-in-production)
10. [Integration Examples](#10-integration-examples)
11. [MCP Ecosystem](#11-mcp-ecosystem)
12. [Q&A Section](#12-qa-section)

---

## 1. MCP Overview

### What is MCP?

**Model Context Protocol (MCP)** is an open protocol created by Anthropic that provides
a standardized way for Large Language Models (LLMs) to interact with external tools,
data sources, and services. Think of it as a **"USB-C for AI applications"** -- a
universal connector that lets any AI model talk to any tool through a single, well-defined
interface.

MCP was open-sourced by Anthropic in November 2024.

### Why MCP Was Created

Before MCP, every AI application had to build custom integrations for each tool or data
source. This created an **M x N integration problem**: if you had M applications and N
tools, you needed M x N custom connectors.

```
THE PROBLEM: M x N Integrations
================================

Without MCP:

  Claude ──────┬── GitHub API      (custom connector)
  Claude ──────┼── Slack API       (custom connector)
  Claude ──────┼── PostgreSQL      (custom connector)
  Claude ──────┼── Filesystem      (custom connector)
               │
  ChatGPT ─────┬── GitHub API      (different custom connector)
  ChatGPT ─────┼── Slack API       (different custom connector)
  ChatGPT ─────┼── PostgreSQL      (different custom connector)
  ChatGPT ─────┼── Filesystem      (different custom connector)
               │
  Custom App ──┬── GitHub API      (yet another connector)
  Custom App ──┼── Slack API       (yet another connector)
  Custom App ──┼── PostgreSQL      (yet another connector)
  Custom App ──┼── Filesystem      (yet another connector)

  Total connections: 3 apps x 4 tools = 12 custom integrations


THE SOLUTION: M + N Integrations
=================================

With MCP:

  Claude ──────┐                  ┌── GitHub MCP Server
  ChatGPT ─────┼── MCP Protocol ──┼── Slack MCP Server
  Custom App ──┘                  ├── PostgreSQL MCP Server
                                  └── Filesystem MCP Server

  Total connections: 3 clients + 4 servers = 7 standardized integrations
```

### Key Benefits

| Benefit                | Description                                             |
|------------------------|---------------------------------------------------------|
| **Standardization**    | One protocol replaces dozens of custom integrations     |
| **Reusability**        | Build a server once, use it with any MCP-compatible app |
| **Composability**      | Combine multiple servers for complex workflows          |
| **Security**           | Controlled access with clear permission boundaries      |
| **Ecosystem**          | Growing library of pre-built servers                    |

### MCP Architecture

MCP uses a **client-server architecture** with three distinct roles:

```
MCP Architecture
=================

┌─────────────────────────────────────────────────────┐
│                      HOST                           │
│            (e.g., Claude Desktop, IDE)              │
│                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  MCP Client │  │  MCP Client │  │  MCP Client │ │
│  │      A      │  │      B      │  │      C      │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
│         │                │                │         │
└─────────┼────────────────┼────────────────┼─────────┘
          │                │                │
    MCP Protocol     MCP Protocol     MCP Protocol
    (JSON-RPC 2.0)   (JSON-RPC 2.0)   (JSON-RPC 2.0)
          │                │                │
   ┌──────┴──────┐  ┌─────┴──────┐  ┌─────┴──────┐
   │ MCP Server  │  │ MCP Server │  │ MCP Server │
   │  (GitHub)   │  │  (Slack)   │  │ (Database) │
   └─────────────┘  └────────────┘  └────────────┘
         │                │                │
    GitHub API       Slack API        PostgreSQL
```

**Three Roles:**

| Role       | Description                                                        |
|------------|--------------------------------------------------------------------|
| **Host**   | The application that contains the LLM (Claude Desktop, IDE, etc.)  |
| **Client** | A connector within the host that maintains a 1:1 session with a server |
| **Server** | A lightweight program that exposes tools, resources, and prompts   |

**Key Architectural Principles:**

- Each client maintains a **1:1 connection** with a single server.
- A host can spawn **multiple clients**, each connecting to a different server.
- The host controls which servers are available and manages security policies.
- Communication uses **JSON-RPC 2.0** as the message format.

### Transport Protocols

MCP supports multiple transport mechanisms:

| Transport            | Use Case                    | Description                           |
|----------------------|-----------------------------|---------------------------------------|
| **stdio**            | Local servers               | Communication via stdin/stdout        |
| **Streamable HTTP**  | Remote/networked servers    | HTTP POST + optional SSE streaming    |
| **SSE (legacy)**     | Remote servers (deprecated) | Server-Sent Events over HTTP          |

---

## 2. Core Concepts

MCP defines four main **primitives** that servers can expose:

```
MCP Primitives
===============

┌─────────────────────────────────────────────────────────────┐
│                        MCP SERVER                           │
│                                                             │
│  ┌─────────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐ │
│  │  Resources  │  │  Tools   │  │ Prompts │  │ Sampling │ │
│  │             │  │          │  │         │  │          │ │
│  │ - Files     │  │ - Search │  │ - Code  │  │ - Server │ │
│  │ - DB rows   │  │ - CRUD   │  │   review│  │   asks   │ │
│  │ - API data  │  │ - Compute│  │ - Debug │  │   LLM to │ │
│  │ - Configs   │  │ - Execute│  │ - Summar│  │   think  │ │
│  │             │  │          │  │   ize   │  │          │ │
│  └─────────────┘  └──────────┘  └─────────┘  └──────────┘ │
│                                                             │
│   Read-only data    Actions &     Reusable     LLM calls   │
│   for context       side effects  templates    from server  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Resources

**Resources** are data that servers expose to clients. They represent any kind of data
the LLM might need as context -- files, database records, API responses, live system data.

- Resources are **read-only** and intended to provide **context**.
- Identified by URIs (e.g., `file:///path/to/doc.txt`, `db://users/123`).
- Can be **static** (fixed list) or **dynamic** (discovered via templates).

```python
# Example: Exposing a resource
@server.list_resources()
async def list_resources():
    return [
        Resource(
            uri="config://app/settings",
            name="Application Settings",
            description="Current application configuration",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def read_resource(uri: str):
    if uri == "config://app/settings":
        settings = load_settings()
        return json.dumps(settings)
```

### Tools

**Tools** are functions that servers expose for the LLM to call. Unlike resources,
tools can perform **actions and have side effects** -- creating files, querying APIs,
modifying databases, etc.

- Tools are **model-controlled**: the LLM decides when and how to use them.
- Each tool has a **name**, **description**, and **input schema** (JSON Schema).
- Tool calls are typically mediated by the host (human-in-the-loop approval).

```python
@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="create_issue",
            description="Create a new GitHub issue",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Issue title"},
                    "body": {"type": "string", "description": "Issue body"},
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Issue labels"
                    }
                },
                "required": ["title"]
            }
        )
    ]
```

### Prompts

**Prompts** are pre-defined prompt templates that servers expose. They allow servers to
offer reusable, parameterized prompt structures for common workflows.

- Prompts are **user-controlled**: the user explicitly selects them.
- They can accept **arguments** for customization.
- They return structured messages (user/assistant roles).

```python
@server.list_prompts()
async def list_prompts():
    return [
        Prompt(
            name="code_review",
            description="Review code for best practices",
            arguments=[
                PromptArgument(
                    name="language",
                    description="Programming language",
                    required=True
                )
            ]
        )
    ]
```

### Sampling

**Sampling** is a unique feature where the **server** can request the host's LLM to
generate text. This enables agentic behaviors where the server itself needs LLM
reasoning.

- Server sends a `sampling/createMessage` request to the client/host.
- The host mediates this (can show the user, modify, approve/reject).
- Enables multi-step agent workflows within MCP servers.

```
Sampling Flow
==============

  ┌────────┐                    ┌────────┐
  │  Host  │                    │ Server │
  │ (LLM)  │                    │        │
  └───┬────┘                    └───┬────┘
      │                             │
      │  1. Client calls tool       │
      │────────────────────────────►│
      │                             │
      │  2. Server needs LLM help   │
      │◄────────────────────────────│
      │   sampling/createMessage    │
      │                             │
      │  3. Host runs LLM,         │
      │     returns result          │
      │────────────────────────────►│
      │                             │
      │  4. Server completes tool   │
      │◄────────────────────────────│
      │   tool result               │
      └────────────────────────────►┘
```

### Control Hierarchy

```
Who Controls What?
===================

  ┌──────────────────────────────────────────────┐
  │              USER (Human)                    │
  │  Controls: Prompt selection, approval        │
  │  ┌──────────────────────────────────────────┐│
  │  │            HOST (Application)            ││
  │  │  Controls: Security, server lifecycle    ││
  │  │  ┌──────────────────────────────────────┐││
  │  │  │           LLM (Model)               │││
  │  │  │  Controls: Tool invocation,         │││
  │  │  │  resource reading decisions         │││
  │  │  └──────────────────────────────────────┘││
  │  └──────────────────────────────────────────┘│
  └──────────────────────────────────────────────┘

  Prompts  --> User-controlled (explicit selection)
  Resources --> Application-controlled (contextual)
  Tools    --> Model-controlled (LLM decides when to call)
```

---

## 3. MCP Server Development

### Setting Up a Python MCP Server

**Installation:**

```bash
# Install the MCP Python SDK
pip install mcp

# Or with optional dependencies for HTTP transport
pip install "mcp[cli]"
```

**Project structure:**

```
my-mcp-server/
├── pyproject.toml
├── src/
│   └── my_mcp_server/
│       ├── __init__.py
│       └── server.py
└── README.md
```

### Complete MCP Server Example

```python
"""
Complete MCP Server: Weather + Notes Service
=============================================
Demonstrates resources, tools, and prompts in a single server.
"""

import json
import asyncio
import logging
from datetime import datetime
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    Prompt,
    PromptMessage,
    PromptArgument,
    GetPromptResult,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-notes-server")

server = Server("weather-notes-server")

# In-memory data store
notes: dict[str, str] = {}

# Simulated weather data
WEATHER_DATA = {
    "new york": {"temp": 22, "condition": "Partly Cloudy", "humidity": 65},
    "london": {"temp": 15, "condition": "Rainy", "humidity": 80},
    "tokyo": {"temp": 28, "condition": "Sunny", "humidity": 45},
    "paris": {"temp": 18, "condition": "Cloudy", "humidity": 70},
}

# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------
@server.list_resources()
async def list_resources() -> list[Resource]:
    """List all available resources."""
    resources = [
        Resource(
            uri="weather://cities",
            name="Available Cities",
            description="List of cities with weather data",
            mimeType="application/json",
        )
    ]
    # Add each note as a resource
    for note_id, content in notes.items():
        resources.append(
            Resource(
                uri=f"note://notes/{note_id}",
                name=f"Note: {note_id}",
                description=f"User note: {note_id}",
                mimeType="text/plain",
            )
        )
    return resources


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read a specific resource by URI."""
    if uri == "weather://cities":
        return json.dumps(list(WEATHER_DATA.keys()))

    if uri.startswith("note://notes/"):
        note_id = uri.replace("note://notes/", "")
        if note_id in notes:
            return notes[note_id]
        raise ValueError(f"Note not found: {note_id}")

    raise ValueError(f"Unknown resource URI: {uri}")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    return [
        Tool(
            name="get_weather",
            description="Get current weather for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (e.g., 'new york', 'london')",
                    }
                },
                "required": ["city"],
            },
        ),
        Tool(
            name="add_note",
            description="Add a new note",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Note title / identifier",
                    },
                    "content": {
                        "type": "string",
                        "description": "Note content",
                    },
                },
                "required": ["title", "content"],
            },
        ),
        Tool(
            name="search_notes",
            description="Search notes by keyword",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    }
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Execute a tool with the given arguments."""
    if name == "get_weather":
        city = arguments["city"].lower()
        if city not in WEATHER_DATA:
            available = ", ".join(WEATHER_DATA.keys())
            return [
                TextContent(
                    type="text",
                    text=f"City '{city}' not found. Available: {available}",
                )
            ]
        data = WEATHER_DATA[city]
        result = (
            f"Weather in {city.title()}:\n"
            f"  Temperature: {data['temp']}C\n"
            f"  Condition:   {data['condition']}\n"
            f"  Humidity:    {data['humidity']}%"
        )
        return [TextContent(type="text", text=result)]

    elif name == "add_note":
        title = arguments["title"]
        content = arguments["content"]
        notes[title] = content
        # Notify clients that the resource list has changed
        await server.request_context.session.send_resource_list_changed()
        return [
            TextContent(type="text", text=f"Note '{title}' added successfully.")
        ]

    elif name == "search_notes":
        query = arguments["query"].lower()
        matches = {
            title: content
            for title, content in notes.items()
            if query in title.lower() or query in content.lower()
        }
        if not matches:
            return [TextContent(type="text", text="No matching notes found.")]
        result_lines = ["Matching notes:"]
        for title, content in matches.items():
            result_lines.append(f"  [{title}]: {content[:100]}")
        return [TextContent(type="text", text="\n".join(result_lines))]

    else:
        raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    """List all available prompt templates."""
    return [
        Prompt(
            name="weather_report",
            description="Generate a detailed weather report for a city",
            arguments=[
                PromptArgument(
                    name="city",
                    description="City name",
                    required=True,
                )
            ],
        ),
        Prompt(
            name="summarize_notes",
            description="Summarize all stored notes",
        ),
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
    """Return a specific prompt with its messages."""
    if name == "weather_report":
        city = (arguments or {}).get("city", "unknown")
        return GetPromptResult(
            description=f"Weather report for {city}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Please provide a comprehensive weather report for "
                            f"{city}. Use the get_weather tool to fetch current "
                            f"data, then provide a detailed analysis including "
                            f"clothing recommendations and activity suggestions."
                        ),
                    ),
                )
            ],
        )

    elif name == "summarize_notes":
        notes_text = "\n".join(
            f"- {title}: {content}" for title, content in notes.items()
        )
        return GetPromptResult(
            description="Summary of all notes",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Please summarize the following notes and identify "
                            f"key themes:\n\n{notes_text or '(No notes yet)'}"
                        ),
                    ),
                )
            ],
        )

    raise ValueError(f"Unknown prompt: {name}")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
async def main():
    """Run the MCP server over stdio."""
    logger.info("Starting Weather + Notes MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
```

### Server Lifecycle

```
MCP Server Lifecycle
=====================

  Client                              Server
    │                                    │
    │  ── initialize ──────────────────► │  Negotiate capabilities
    │  ◄── initialize response ──────── │  (protocol version,
    │                                    │   supported features)
    │  ── initialized notification ───► │
    │                                    │
    │         Normal Operation           │
    │  ── list_tools ──────────────────► │
    │  ◄── tools list ─────────────────  │
    │  ── call_tool ───────────────────► │
    │  ◄── tool result ────────────────  │
    │  ── list_resources ──────────────► │
    │  ◄── resources list ─────────────  │
    │  ── read_resource ───────────────► │
    │  ◄── resource content ───────────  │
    │                                    │
    │         Shutdown                   │
    │  ── shutdown (or close transport)  │
    │                                    │
```

**Initialization handshake:**

```python
# The client sends an 'initialize' request with:
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {
            "sampling": {}          # Client supports sampling
        },
        "clientInfo": {
            "name": "my-client",
            "version": "1.0.0"
        }
    }
}

# The server responds with:
{
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "protocolVersion": "2025-03-26",
        "capabilities": {
            "tools": {},            # Server offers tools
            "resources": {
                "subscribe": True   # Server supports subscriptions
            },
            "prompts": {}           # Server offers prompts
        },
        "serverInfo": {
            "name": "weather-notes-server",
            "version": "1.0.0"
        }
    }
}
```

### Using the FastMCP High-Level API

The `mcp` Python SDK also provides a higher-level decorator-based API called
**FastMCP** that simplifies server development considerably:

```python
"""
FastMCP -- High-Level Server API
=================================
Reduced boilerplate compared to the low-level Server class.
"""

from mcp.server.fastmcp import FastMCP

# Create server
mcp = FastMCP("Demo Server")


# --- Tools (decorated functions become tools automatically) ----------------
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@mcp.tool()
def get_weather(city: str, units: str = "celsius") -> str:
    """Get current weather for a city.

    Args:
        city: Name of the city.
        units: Temperature units -- 'celsius' or 'fahrenheit'.
    """
    # In production, call a real weather API
    return f"Weather in {city}: 22 degrees {units}, sunny"


# --- Resources (expose data via URIs) -------------------------------------
@mcp.resource("config://app")
def get_config() -> str:
    """Return application configuration."""
    return json.dumps({"debug": False, "version": "2.0"})


@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    """Return user profile by ID (dynamic resource template)."""
    return json.dumps({"id": user_id, "name": "Alice", "role": "admin"})


# --- Prompts ---------------------------------------------------------------
@mcp.prompt()
def review_code(code: str, language: str = "python") -> str:
    """Generate a code review prompt."""
    return f"Please review this {language} code:\n\n```{language}\n{code}\n```"


# --- Run -------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()          # Defaults to stdio transport
    # mcp.run(transport="sse")   # For HTTP/SSE transport
```

**FastMCP advantages:**

- Automatic JSON Schema generation from Python type hints.
- Docstrings become tool/resource descriptions.
- Less boilerplate -- no manual `list_tools` / `call_tool` handlers.
- Built-in argument validation.

---

## 4. MCP Client Development

### Building a Python MCP Client

```python
"""
MCP Client Example
===================
Connects to an MCP server, discovers tools, and calls them.
"""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # 1. Define the server to connect to
    server_params = StdioServerParameters(
        command="python",
        args=["path/to/my_mcp_server.py"],
        env=None,  # Optional environment variables
    )

    # 2. Connect to the server via stdio
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 3. Initialize the connection
            await session.initialize()

            # 4. Discover available tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            # 5. Discover available resources
            resources = await session.list_resources()
            print("\nAvailable resources:")
            for resource in resources.resources:
                print(f"  - {resource.uri}: {resource.name}")

            # 6. Call a tool
            result = await session.call_tool(
                "get_weather",
                arguments={"city": "london"},
            )
            print(f"\nWeather result: {result.content[0].text}")

            # 7. Read a resource
            resource_content = await session.read_resource("weather://cities")
            print(f"\nCities: {resource_content.contents[0].text}")

            # 8. Get a prompt
            prompts = await session.list_prompts()
            print("\nAvailable prompts:")
            for prompt in prompts.prompts:
                print(f"  - {prompt.name}: {prompt.description}")

            prompt_result = await session.get_prompt(
                "weather_report",
                arguments={"city": "Tokyo"},
            )
            print(f"\nPrompt: {prompt_result.messages[0].content.text}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Client with HTTP/SSE Transport

```python
"""
MCP Client over HTTP/SSE
=========================
Connects to a remote MCP server via Streamable HTTP.
"""

import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main():
    # Connect to a remote MCP server over HTTP
    async with streamablehttp_client(
        url="http://localhost:8000/mcp"
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"Tool: {tool.name} -- {tool.description}")

            result = await session.call_tool(
                "get_weather",
                arguments={"city": "paris"},
            )
            print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
```

### Multi-Server Client

```python
"""
Multi-Server MCP Client
=========================
Connects to multiple MCP servers simultaneously.
"""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MultiServerClient:
    """Manages connections to multiple MCP servers."""

    def __init__(self):
        self.sessions: dict[str, ClientSession] = {}
        self.tools: dict[str, tuple[str, dict]] = {}   # tool_name -> (server, schema)

    async def connect_server(self, name: str, command: str, args: list[str]):
        """Connect to a server and register its tools."""
        server_params = StdioServerParameters(command=command, args=args)

        # NOTE: In production, you'd manage these context managers
        # properly with a lifecycle manager.
        read, write = await stdio_client(server_params).__aenter__()
        session = await ClientSession(read, write).__aenter__()
        await session.initialize()

        self.sessions[name] = session

        # Discover and register tools
        tools = await session.list_tools()
        for tool in tools.tools:
            self.tools[tool.name] = (name, tool.inputSchema)
            print(f"Registered tool '{tool.name}' from server '{name}'")

    async def call_tool(self, tool_name: str, arguments: dict):
        """Route a tool call to the correct server."""
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        server_name, _ = self.tools[tool_name]
        session = self.sessions[server_name]
        return await session.call_tool(tool_name, arguments=arguments)

    def get_all_tool_schemas(self) -> list[dict]:
        """Return schemas for all tools across all servers (for LLM)."""
        schemas = []
        for tool_name, (server_name, schema) in self.tools.items():
            schemas.append({
                "name": tool_name,
                "server": server_name,
                "inputSchema": schema,
            })
        return schemas
```

---

## 5. Resources in Detail

### Static vs Dynamic Resources

```
Resource Types
===============

Static Resources                    Dynamic Resources (Templates)
─────────────────                   ───────────────────────────────
  Known at startup                    Generated from URI templates
  Fixed URI                           URI with placeholders
  Always in list_resources()          Discovered via resource_templates

  Example:                            Example:
  "config://app/settings"             "db://users/{user_id}/profile"
  "docs://readme"                     "logs://{date}/errors"
```

### Resource Templates

Resource templates use **URI templates** (RFC 6570) to describe dynamic resources:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Resource Demo")


# Static resource -- always listed
@mcp.resource("system://status")
def get_system_status() -> str:
    """Return current system status."""
    return json.dumps({
        "status": "healthy",
        "uptime": "48h",
        "version": "2.1.0",
    })


# Dynamic resource template -- URI with parameters
@mcp.resource("db://users/{user_id}")
def get_user(user_id: str) -> str:
    """Fetch a user by ID."""
    # In production, query actual database
    return json.dumps({
        "id": user_id,
        "name": "Alice",
        "email": "alice@example.com",
    })


# Dynamic resource with multiple parameters
@mcp.resource("logs://{service}/{date}")
def get_logs(service: str, date: str) -> str:
    """Fetch logs for a service on a given date."""
    return json.dumps({
        "service": service,
        "date": date,
        "entries": [
            {"level": "INFO", "msg": "Service started"},
            {"level": "ERROR", "msg": "Connection timeout"},
        ],
    })
```

### Resource Subscriptions

Servers can notify clients when resource content changes:

```python
from mcp.server import Server
from mcp.types import Resource

server = Server("subscription-demo")

# Track subscribed resources
subscribed_resources: set[str] = set()


@server.subscribe_resource()
async def subscribe(uri: str):
    """Handle a client subscribing to resource changes."""
    subscribed_resources.add(uri)


@server.unsubscribe_resource()
async def unsubscribe(uri: str):
    """Handle a client unsubscribing."""
    subscribed_resources.discard(uri)


async def notify_resource_change(uri: str):
    """
    Call this whenever a resource's content changes.
    The client will re-read the resource.
    """
    if uri in subscribed_resources:
        await server.request_context.session.send_resource_updated(uri=uri)
```

### Resource MIME Types

Resources specify their content type via MIME types:

| MIME Type                | Use Case                        |
|--------------------------|---------------------------------|
| `text/plain`             | Plain text content              |
| `application/json`       | JSON data                       |
| `text/markdown`          | Markdown documents              |
| `text/html`              | HTML content                    |
| `application/octet-stream` | Binary data (base64 encoded) |
| `image/png`              | Images (base64 encoded)         |

```python
@server.list_resources()
async def list_resources():
    return [
        Resource(
            uri="report://monthly",
            name="Monthly Report",
            mimeType="text/markdown",     # Markdown content
        ),
        Resource(
            uri="chart://sales",
            name="Sales Chart",
            mimeType="image/png",         # Binary image
        ),
    ]

@server.read_resource()
async def read_resource(uri: str):
    if uri == "report://monthly":
        return "# Monthly Report\n\nRevenue increased by 15%..."

    if uri == "chart://sales":
        # Return base64-encoded binary content
        import base64
        with open("sales_chart.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
```

---

## 6. Tools in Detail

### Tool Input Schemas

Tools use **JSON Schema** to define their expected input:

```python
Tool(
    name="query_database",
    description="Execute a read-only SQL query against the database",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL SELECT query to execute",
            },
            "database": {
                "type": "string",
                "enum": ["users", "orders", "products"],
                "description": "Target database",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1000,
                "default": 100,
                "description": "Maximum number of rows to return",
            },
        },
        "required": ["query", "database"],
    },
)
```

### Error Handling in Tools

```python
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls with proper error handling."""
    try:
        if name == "query_database":
            return await handle_query(arguments)
        elif name == "create_file":
            return await handle_create_file(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    except ValueError as e:
        # Return error as text content -- not an exception.
        # The LLM will see this message and can adjust.
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}",
            isError=True,        # Marks this as an error result
        )]

    except PermissionError as e:
        return [TextContent(
            type="text",
            text=f"Permission denied: {str(e)}",
            isError=True,
        )]

    except Exception as e:
        logger.exception(f"Unexpected error in tool '{name}'")
        return [TextContent(
            type="text",
            text=f"Internal error: {str(e)}",
            isError=True,
        )]
```

### Tool Annotations

Tools can include **annotations** that provide metadata about their behavior,
helping hosts decide on confirmation policies:

```python
Tool(
    name="delete_file",
    description="Delete a file from the filesystem",
    inputSchema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to delete"}
        },
        "required": ["path"],
    },
    annotations={
        "title": "Delete File",
        "readOnlyHint": False,        # This tool modifies state
        "destructiveHint": True,      # This tool is destructive
        "idempotentHint": True,       # Calling twice has same effect
        "openWorldHint": False,       # Only accesses local filesystem
    },
)
```

**Annotation hints:**

| Hint               | Description                                             |
|--------------------|---------------------------------------------------------|
| `readOnlyHint`     | True if the tool does not modify anything               |
| `destructiveHint`  | True if the tool performs destructive/irreversible actions |
| `idempotentHint`   | True if calling the tool multiple times has the same result |
| `openWorldHint`    | True if the tool interacts with external systems        |

### Progress Reporting

For long-running tools, servers can report progress:

```python
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "bulk_import":
        items = arguments["items"]
        total = len(items)
        results = []

        for i, item in enumerate(items):
            # Report progress via progress token
            ctx = server.request_context
            if ctx.meta and ctx.meta.progressToken:
                await ctx.session.send_progress_notification(
                    progress_token=ctx.meta.progressToken,
                    progress=i + 1,
                    total=total,
                    message=f"Processing item {i + 1}/{total}",
                )
            results.append(await process_item(item))

        return [TextContent(
            type="text",
            text=f"Imported {total} items successfully.",
        )]
```

---

## 7. Prompts in Detail

### Prompt Templates with Arguments

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Prompt Demo")


@mcp.prompt()
def debug_error(
    error_message: str,
    stack_trace: str = "",
    language: str = "python",
) -> str:
    """Generate a debugging prompt for an error."""
    prompt = f"I'm encountering the following error in my {language} code:\n\n"
    prompt += f"Error: {error_message}\n"
    if stack_trace:
        prompt += f"\nStack trace:\n```\n{stack_trace}\n```\n"
    prompt += "\nPlease help me:\n"
    prompt += "1. Understand what caused this error\n"
    prompt += "2. Suggest a fix\n"
    prompt += "3. Explain how to prevent it in the future"
    return prompt
```

### Multi-Message Prompts

Prompts can return multiple messages with different roles to set up a
conversation context:

```python
from mcp.types import (
    GetPromptResult,
    PromptMessage,
    TextContent,
    EmbeddedResource,
    ResourceContents,
    TextResourceContents,
)


@server.get_prompt()
async def get_prompt(
    name: str, arguments: dict[str, str] | None
) -> GetPromptResult:
    if name == "code_review_session":
        code = (arguments or {}).get("code", "")
        language = (arguments or {}).get("language", "python")

        return GetPromptResult(
            description="Interactive code review session",
            messages=[
                # Set up assistant context
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            "You are a senior software engineer performing "
                            "a thorough code review. Focus on: correctness, "
                            "performance, security, and maintainability."
                        ),
                    ),
                ),
                PromptMessage(
                    role="assistant",
                    content=TextContent(
                        type="text",
                        text=(
                            "I'll review the code carefully, examining each "
                            "aspect systematically. Please share the code."
                        ),
                    ),
                ),
                # Provide the code to review
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"Please review this {language} code:\n\n```{language}\n{code}\n```",
                    ),
                ),
            ],
        )
```

### Prompts with Embedded Resources

Prompts can include references to server resources:

```python
@server.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
    if name == "analyze_logs":
        service = (arguments or {}).get("service", "api")
        return GetPromptResult(
            description=f"Analyze logs for {service}",
            messages=[
                PromptMessage(
                    role="user",
                    content=EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri=f"logs://{service}/today",
                            mimeType="text/plain",
                            text="",   # Client will read actual content
                        ),
                    ),
                ),
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            "Analyze the above logs and identify:\n"
                            "1. Error patterns\n"
                            "2. Performance bottlenecks\n"
                            "3. Security concerns\n"
                            "4. Recommendations"
                        ),
                    ),
                ),
            ],
        )
```

---

## 8. Transport Protocols

### stdio Transport

**Best for:** Local servers, CLI tools, development, desktop applications.

```
stdio Transport
================

  ┌──────────┐  stdin (JSON-RPC)   ┌──────────┐
  │  Client  │ ──────────────────► │  Server  │
  │          │                     │ (child   │
  │          │ ◄────────────────── │  process)│
  └──────────┘  stdout (JSON-RPC)  └──────────┘

  - Server runs as a child process of the client
  - Messages sent via stdin/stdout
  - stderr used for logging (not protocol messages)
  - Simple, secure, no network exposure
```

**Server-side (stdio):**

```python
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("my-local-server")

# ... define tools, resources, prompts ...

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
```

**Client-side (stdio):**

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python",
    args=["my_server.py"],
    env={"API_KEY": "secret"},   # Environment variables for the server
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        # ... use session ...
```

### Streamable HTTP Transport

**Best for:** Remote servers, cloud deployments, multi-client scenarios.

The **Streamable HTTP** transport (introduced in MCP protocol version 2025-03-26)
replaces the earlier SSE transport. It uses a single HTTP endpoint that accepts
POST requests and can optionally upgrade responses to Server-Sent Events for
streaming.

```
Streamable HTTP Transport
==========================

  ┌──────────┐                            ┌──────────┐
  │  Client  │  POST /mcp (JSON-RPC)      │  Server  │
  │          │ ──────────────────────────► │  (HTTP)  │
  │          │                            │          │
  │          │ ◄────────────────────────── │          │
  │          │  Response: JSON or SSE      │          │
  └──────────┘                            └──────────┘

  - Single endpoint (e.g., /mcp)
  - Client sends JSON-RPC via HTTP POST
  - Server responds with:
    - Direct JSON response (simple request/response), or
    - SSE stream (for streaming results / notifications)
  - Supports session management via Mcp-Session-Id header
  - Stateful or stateless operation
```

**Server-side (Streamable HTTP):**

```python
"""
MCP Server with Streamable HTTP transport
==========================================
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Remote Server")


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)   # In production, use a safe evaluator!
        return str(result)
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
```

**Client-side (Streamable HTTP):**

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client(
    url="http://localhost:8000/mcp"
) as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool(
            "calculate", arguments={"expression": "2 ** 10"}
        )
        print(result.content[0].text)  # "1024"
```

### SSE Transport (Legacy)

The original SSE transport used two endpoints and is now deprecated in favor of
Streamable HTTP:

```
Legacy SSE Transport (Deprecated)
==================================

  ┌──────────┐  GET /sse (event stream)   ┌──────────┐
  │  Client  │ ◄────────────────────────── │  Server  │
  │          │                            │          │
  │          │  POST /messages (JSON-RPC)  │          │
  │          │ ──────────────────────────► │          │
  └──────────┘                            └──────────┘

  - Two endpoints required
  - Client opens SSE connection for server messages
  - Client posts requests to /messages
```

### Transport Comparison

| Feature              | stdio           | Streamable HTTP     | SSE (Legacy)      |
|----------------------|-----------------|---------------------|--------------------|
| Deployment           | Local only      | Local or remote     | Local or remote    |
| Multiple clients     | No (1:1)        | Yes                 | Yes                |
| Network exposure     | None            | HTTP port           | HTTP port          |
| Session management   | Implicit        | Via headers         | Via endpoint URLs  |
| Streaming            | N/A             | Optional SSE        | Always SSE         |
| Authentication       | OS-level        | HTTP auth/tokens    | HTTP auth/tokens   |
| Best for             | Desktop apps    | Cloud/production    | (deprecated)       |

---

## 9. MCP in Production

### Server Deployment Patterns

```
Deployment Architectures
=========================

Pattern 1: Sidecar (Local)
───────────────────────────
  ┌─────────────────────────────┐
  │  Application Host           │
  │  ┌────────┐  ┌────────────┐ │
  │  │ Client │──│ MCP Server │ │    stdio
  │  └────────┘  │ (sidecar)  │ │
  │              └──────┬─────┘ │
  └─────────────────────┼───────┘
                        │
                   Local DB / Files


Pattern 2: Standalone Remote Service
──────────────────────────────────────
  ┌──────────┐                    ┌──────────────┐
  │ Client A │───── HTTPS ──────► │  MCP Server  │
  └──────────┘                    │  (deployed)  │
  ┌──────────┐                    │              │
  │ Client B │───── HTTPS ──────► │              │
  └──────────┘                    └──────┬───────┘
                                         │
                                   Cloud APIs / DB


Pattern 3: Gateway (Multiple Backends)
───────────────────────────────────────
  ┌──────────┐       ┌───────────────┐     ┌──────────┐
  │  Client  │──────►│  MCP Gateway  │────►│ Service A│
  └──────────┘       │  (aggregator) │────►│ Service B│
                     └───────────────┘────►│ Service C│
                                           └──────────┘
```

### Security Considerations

```python
"""
Security Best Practices for MCP Servers
=========================================
"""

# 1. Input Validation
# ---------------------
import re
from pathlib import Path

ALLOWED_PATHS = [Path("/data"), Path("/tmp/mcp")]

def validate_file_path(path_str: str) -> Path:
    """Validate and sanitize file paths."""
    path = Path(path_str).resolve()

    # Prevent path traversal attacks
    if not any(path.is_relative_to(allowed) for allowed in ALLOWED_PATHS):
        raise PermissionError(
            f"Access denied: {path} is outside allowed directories"
        )
    return path


def validate_sql_query(query: str) -> str:
    """Validate SQL queries -- allow only SELECT statements."""
    normalized = query.strip().upper()
    if not normalized.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")

    # Block dangerous keywords
    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "EXEC", "GRANT"]
    for keyword in dangerous:
        if re.search(rf'\b{keyword}\b', normalized):
            raise ValueError(f"Forbidden keyword: {keyword}")

    return query


# 2. Authentication (HTTP transport)
# ------------------------------------
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware


class TokenAuthBackend:
    """Simple token-based authentication."""

    async def authenticate(self, request):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            raise AuthenticationError("Missing token")
        # Validate token against your auth service
        user = await validate_token(token)
        return user


# 3. Rate Limiting
# ------------------
import time
from collections import defaultdict

class RateLimiter:
    """Simple in-memory rate limiter for tool calls."""

    def __init__(self, max_calls: int = 100, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window = window_seconds
        self.calls: dict[str, list[float]] = defaultdict(list)

    def check(self, client_id: str) -> bool:
        now = time.time()
        # Remove old entries
        self.calls[client_id] = [
            t for t in self.calls[client_id]
            if now - t < self.window
        ]
        if len(self.calls[client_id]) >= self.max_calls:
            return False
        self.calls[client_id].append(now)
        return True

rate_limiter = RateLimiter(max_calls=100, window_seconds=60)


# 4. Logging and Audit Trail
# ----------------------------
import structlog

logger = structlog.get_logger()

async def audited_tool_call(name: str, arguments: dict, client_id: str):
    """Wrap tool calls with audit logging."""
    logger.info(
        "tool_call_started",
        tool=name,
        client=client_id,
        arguments=arguments,
    )
    try:
        result = await execute_tool(name, arguments)
        logger.info(
            "tool_call_completed",
            tool=name,
            client=client_id,
            success=True,
        )
        return result
    except Exception as e:
        logger.error(
            "tool_call_failed",
            tool=name,
            client=client_id,
            error=str(e),
        )
        raise
```

### Error Handling Patterns

```python
"""
Robust Error Handling for MCP Servers
======================================
"""

from mcp.types import TextContent, INTERNAL_ERROR, INVALID_PARAMS
from mcp.shared.exceptions import McpError


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Central tool dispatcher with layered error handling."""

    # Layer 1: Tool-level errors (expected)
    # Return as text content with isError=True
    try:
        handler = TOOL_HANDLERS.get(name)
        if not handler:
            return [TextContent(type="text", text=f"Unknown tool: {name}", isError=True)]

        result = await handler(arguments)
        return [TextContent(type="text", text=result)]

    except ValueError as e:
        # User/input errors -- LLM can try again with different args
        return [TextContent(type="text", text=f"Invalid input: {e}", isError=True)]

    except PermissionError as e:
        # Access control errors
        return [TextContent(type="text", text=f"Access denied: {e}", isError=True)]

    except TimeoutError:
        return [TextContent(
            type="text",
            text="Request timed out. Please try again or simplify your query.",
            isError=True,
        )]

    except Exception as e:
        # Layer 2: Protocol-level errors (unexpected)
        # Raise McpError for protocol-level issues
        logger.exception(f"Unexpected error in tool '{name}'")
        raise McpError(INTERNAL_ERROR, f"Internal server error: {type(e).__name__}")
```

### Configuration Management

```python
"""
MCP Server Configuration
==========================
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class MCPServerConfig(BaseSettings):
    """Configuration for an MCP server, loaded from environment variables."""

    # Server identity
    server_name: str = Field(default="my-mcp-server")
    server_version: str = Field(default="1.0.0")

    # Database
    database_url: str = Field(default="postgresql://localhost/mydb")
    db_pool_size: int = Field(default=5)

    # Security
    api_key: str = Field(default="")
    allowed_origins: list[str] = Field(default=["*"])
    max_query_rows: int = Field(default=1000)

    # Transport
    transport: str = Field(default="stdio")        # "stdio" or "streamable-http"
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # Rate limiting
    rate_limit_calls: int = Field(default=100)
    rate_limit_window: int = Field(default=60)

    class Config:
        env_prefix = "MCP_"   # e.g., MCP_DATABASE_URL, MCP_PORT


config = MCPServerConfig()
```

---

## 10. Integration Examples

### MCP + Claude Desktop

Configure Claude Desktop to use MCP servers via `claude_desktop_config.json`:

```json
{
    "mcpServers": {
        "weather": {
            "command": "python",
            "args": ["/path/to/weather_server.py"],
            "env": {
                "API_KEY": "your-weather-api-key"
            }
        },
        "database": {
            "command": "python",
            "args": ["/path/to/db_server.py"],
            "env": {
                "DATABASE_URL": "postgresql://localhost/mydb"
            }
        },
        "remote-service": {
            "url": "https://mcp.example.com/sse",
            "headers": {
                "Authorization": "Bearer token123"
            }
        }
    }
}
```

### MCP + LangChain / LangGraph

```python
"""
Using MCP tools within a LangGraph agent
==========================================
"""

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent


async def main():
    # 1. Connect to MCP servers and get LangChain-compatible tools
    async with MultiServerMCPClient(
        {
            "weather": {
                "command": "python",
                "args": ["weather_server.py"],
                "transport": "stdio",
            },
            "database": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            },
        }
    ) as mcp_client:
        # 2. Get all tools from all connected MCP servers
        tools = mcp_client.get_tools()

        # 3. Create a LangGraph ReAct agent with MCP tools
        model = ChatAnthropic(model="claude-sonnet-4-20250514")
        agent = create_react_agent(model, tools)

        # 4. Run the agent
        result = await agent.ainvoke({
            "messages": [
                {"role": "user", "content": "What's the weather in Tokyo?"}
            ]
        })

        print(result["messages"][-1].content)


asyncio.run(main())
```

### Database MCP Server

```python
"""
Database MCP Server
====================
Exposes a PostgreSQL database via MCP.
"""

import asyncpg
import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("PostgreSQL Server")

# Connection pool (initialized on startup)
pool: asyncpg.Pool | None = None


@mcp.tool()
async def query(sql: str, params: list | None = None) -> str:
    """Execute a read-only SQL query.

    Args:
        sql: A SELECT query to execute.
        params: Optional list of query parameters.
    """
    global pool
    if pool is None:
        pool = await asyncpg.create_pool("postgresql://localhost/mydb")

    # Security: Only allow SELECT
    if not sql.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed."

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *(params or []))
        return json.dumps([dict(row) for row in rows], default=str)


@mcp.tool()
async def list_tables() -> str:
    """List all tables in the database."""
    global pool
    if pool is None:
        pool = await asyncpg.create_pool("postgresql://localhost/mydb")

    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        return json.dumps([row["table_name"] for row in rows])


@mcp.tool()
async def describe_table(table_name: str) -> str:
    """Get the schema of a database table.

    Args:
        table_name: Name of the table to describe.
    """
    global pool
    if pool is None:
        pool = await asyncpg.create_pool("postgresql://localhost/mydb")

    # Prevent SQL injection by parameterizing
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = $1 AND table_schema = 'public'
            ORDER BY ordinal_position
        """, table_name)
        return json.dumps([dict(row) for row in rows])


@mcp.resource("schema://tables")
async def get_all_tables() -> str:
    """Provide a resource listing all tables and their columns."""
    global pool
    if pool is None:
        pool = await asyncpg.create_pool("postgresql://localhost/mydb")

    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
        """)

        tables: dict[str, list] = {}
        for row in rows:
            tables.setdefault(row["table_name"], []).append({
                "column": row["column_name"],
                "type": row["data_type"],
            })
        return json.dumps(tables, indent=2)


if __name__ == "__main__":
    mcp.run()
```

### API Gateway MCP Server

```python
"""
REST API Gateway MCP Server
=============================
Wraps a REST API and exposes it as MCP tools.
"""

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("API Gateway")

BASE_URL = "https://api.example.com/v1"
API_KEY = "your-api-key"


async def api_request(method: str, endpoint: str, data: dict | None = None) -> dict:
    """Make an authenticated API request."""
    async with httpx.AsyncClient() as client:
        response = await client.request(
            method=method,
            url=f"{BASE_URL}{endpoint}",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=data,
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def list_users(page: int = 1, per_page: int = 20) -> str:
    """List users from the API.

    Args:
        page: Page number (starts at 1).
        per_page: Number of users per page.
    """
    data = await api_request("GET", f"/users?page={page}&per_page={per_page}")
    return json.dumps(data, indent=2)


@mcp.tool()
async def get_user(user_id: str) -> str:
    """Get a specific user by ID.

    Args:
        user_id: The user's unique identifier.
    """
    data = await api_request("GET", f"/users/{user_id}")
    return json.dumps(data, indent=2)


@mcp.tool()
async def create_user(name: str, email: str, role: str = "user") -> str:
    """Create a new user.

    Args:
        name: User's full name.
        email: User's email address.
        role: User role (user, admin, moderator).
    """
    data = await api_request("POST", "/users", {
        "name": name,
        "email": email,
        "role": role,
    })
    return json.dumps(data, indent=2)


if __name__ == "__main__":
    mcp.run()
```

### File System MCP Server

```python
"""
File System MCP Server
========================
Provides controlled access to the local filesystem.
"""

import os
import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Filesystem Server")

# Restrict access to a specific base directory
BASE_DIR = Path(os.environ.get("MCP_BASE_DIR", "/tmp/mcp-workspace")).resolve()


def safe_path(relative_path: str) -> Path:
    """Resolve a path safely within the allowed base directory."""
    full_path = (BASE_DIR / relative_path).resolve()
    if not full_path.is_relative_to(BASE_DIR):
        raise PermissionError(f"Access denied: path is outside {BASE_DIR}")
    return full_path


@mcp.tool()
def read_file(path: str) -> str:
    """Read the contents of a file.

    Args:
        path: Relative path within the workspace.
    """
    file_path = safe_path(path)
    if not file_path.exists():
        return f"Error: File not found: {path}"
    if not file_path.is_file():
        return f"Error: Not a file: {path}"
    return file_path.read_text(encoding="utf-8")


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write content to a file (creates parent directories if needed).

    Args:
        path: Relative path within the workspace.
        content: Content to write.
    """
    file_path = safe_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return f"Successfully wrote {len(content)} characters to {path}"


@mcp.tool()
def list_directory(path: str = ".") -> str:
    """List contents of a directory.

    Args:
        path: Relative path within the workspace (default: root).
    """
    dir_path = safe_path(path)
    if not dir_path.is_dir():
        return f"Error: Not a directory: {path}"

    entries = []
    for entry in sorted(dir_path.iterdir()):
        kind = "dir" if entry.is_dir() else "file"
        size = entry.stat().st_size if entry.is_file() else 0
        entries.append({"name": entry.name, "type": kind, "size": size})

    return json.dumps(entries, indent=2)


@mcp.resource("workspace://tree")
def workspace_tree() -> str:
    """Return a tree view of the workspace."""
    lines = []

    def _walk(directory: Path, prefix: str = ""):
        entries = sorted(directory.iterdir())
        for i, entry in enumerate(entries):
            connector = "+-" if i < len(entries) - 1 else "\\-"
            lines.append(f"{prefix}{connector} {entry.name}")
            if entry.is_dir():
                extension = "|  " if i < len(entries) - 1 else "   "
                _walk(entry, prefix + extension)

    _walk(BASE_DIR)
    return "\n".join(lines) if lines else "(empty workspace)"


if __name__ == "__main__":
    mcp.run()
```

---

## 11. MCP Ecosystem

### Popular MCP Servers

| Server               | Description                                         |
|----------------------|-----------------------------------------------------|
| **Filesystem**       | Read/write files with controlled directory access    |
| **GitHub**           | Repository management, issues, PRs, code search     |
| **GitLab**           | GitLab API integration                              |
| **Slack**            | Channel messaging, search, user management          |
| **Google Drive**     | File listing, reading, searching Google Drive        |
| **PostgreSQL**       | Read-only query access to PostgreSQL databases       |
| **SQLite**           | Local SQLite database operations                     |
| **Puppeteer**        | Browser automation, screenshots, page interaction    |
| **Brave Search**     | Web and local search via Brave Search API            |
| **Fetch**            | HTTP requests to fetch web content                   |
| **Memory**           | Persistent knowledge graph for long-term memory      |
| **Sequential Thinking** | Structured step-by-step reasoning                |
| **Docker**           | Container management and execution                   |
| **Kubernetes**       | Cluster management and monitoring                    |

### MCP Server Discovery

Servers can be discovered through:

1. **Anthropic's MCP Server Registry** -- curated list of official and community servers.
2. **npm / PyPI** -- published as packages (`@modelcontextprotocol/server-*` on npm,
   `mcp-server-*` on PyPI).
3. **GitHub repositories** -- many open-source servers on GitHub.
4. **MCP Hub / Directories** -- community-maintained directories of available servers.

### Building Custom Servers for Enterprise

```python
"""
Enterprise MCP Server Pattern
===============================
Demonstrates patterns common in enterprise deployments.
"""

from mcp.server.fastmcp import FastMCP
from contextlib import asynccontextmanager
from dataclasses import dataclass
import asyncpg
import httpx


@dataclass
class AppContext:
    """Shared application context for the MCP server."""
    db_pool: asyncpg.Pool
    http_client: httpx.AsyncClient
    config: dict


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Manage application lifecycle -- setup and teardown."""
    # Startup
    db_pool = await asyncpg.create_pool(
        "postgresql://localhost/enterprise_db",
        min_size=2,
        max_size=10,
    )
    http_client = httpx.AsyncClient(timeout=30.0)
    config = load_config()

    ctx = AppContext(
        db_pool=db_pool,
        http_client=http_client,
        config=config,
    )

    try:
        yield ctx
    finally:
        # Shutdown
        await db_pool.close()
        await http_client.aclose()


mcp = FastMCP("Enterprise Server", lifespan=app_lifespan)


@mcp.tool()
async def get_customer(customer_id: str, ctx: AppContext) -> str:
    """Look up a customer by ID.

    Args:
        customer_id: The customer's unique identifier.
    """
    async with ctx.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM customers WHERE id = $1",
            customer_id,
        )
        if not row:
            return f"Customer {customer_id} not found."
        return json.dumps(dict(row), default=str, indent=2)


@mcp.tool()
async def check_inventory(product_id: str, ctx: AppContext) -> str:
    """Check inventory levels for a product.

    Args:
        product_id: The product's SKU or ID.
    """
    response = await ctx.http_client.get(
        f"{ctx.config['inventory_api']}/products/{product_id}/stock"
    )
    response.raise_for_status()
    return json.dumps(response.json(), indent=2)
```

### Architecture for Multi-Tenant MCP

```
Multi-Tenant MCP Architecture
===============================

  ┌────────────────────────────────────────────────────┐
  │                Load Balancer / API Gateway          │
  │          (Authentication, Rate Limiting)            │
  └──────────┬──────────┬──────────┬───────────────────┘
             │          │          │
  ┌──────────▼──┐ ┌─────▼──────┐ ┌▼─────────────┐
  │ MCP Server  │ │ MCP Server │ │  MCP Server  │
  │ Instance 1  │ │ Instance 2 │ │  Instance N  │
  │ (Tenant A)  │ │ (Tenant B) │ │  (Tenant C)  │
  └──────┬──────┘ └─────┬──────┘ └──────┬───────┘
         │              │               │
  ┌──────▼──────────────▼───────────────▼───────┐
  │              Shared Infrastructure           │
  │   ┌──────────┐  ┌────────┐  ┌────────────┐  │
  │   │ Database │  │ Cache  │  │ Event Bus  │  │
  │   │ (per-    │  │(Redis) │  │ (Kafka)    │  │
  │   │ tenant   │  │        │  │            │  │
  │   │ schemas) │  │        │  │            │  │
  │   └──────────┘  └────────┘  └────────────┘  │
  └──────────────────────────────────────────────┘
```

---

## 12. Q&A Section

### Q1: What is MCP and why was it created?

**A:** MCP (Model Context Protocol) is an open protocol created by Anthropic that
provides a standardized way for LLMs to interact with external tools, data sources, and
services. It was created to solve the **M x N integration problem**: without MCP, if you
have M applications and N tools, you need M x N custom connectors. MCP reduces this to
M + N by providing a universal protocol. Each application only needs to implement the MCP
client protocol, and each tool only needs to implement the MCP server protocol, and they
can all interoperate.

---

### Q2: Explain the MCP architecture (Host, Client, Server).

**A:** MCP has three architectural roles:

- **Host**: The application that contains the LLM (e.g., Claude Desktop, an IDE, a
  custom AI application). The host manages security, user consent, and server lifecycle.
- **Client**: A component within the host that maintains a 1:1 connection with a single
  MCP server. A host can contain multiple clients.
- **Server**: A lightweight program that exposes tools, resources, and prompts via the
  MCP protocol. Servers run as separate processes (stdio) or as HTTP services.

The key insight is that the host creates one client per server, establishing isolated 1:1
sessions, while the host itself orchestrates across all connected servers.

---

### Q3: What are the three main primitives in MCP?

**A:** The three main primitives are:

1. **Resources** -- Data that servers expose for context. Read-only, identified by URIs
   (e.g., `file:///data.json`). The application or user decides when to include them in
   the LLM context.
2. **Tools** -- Functions the server exposes for the LLM to call. Tools can have side
   effects (write files, call APIs). The LLM decides when to invoke them, but the host
   typically asks the user for approval.
3. **Prompts** -- Pre-defined prompt templates that servers expose. The user explicitly
   selects them. They can accept arguments and return structured multi-message
   conversations.

There is also a fourth concept, **Sampling**, where the server can request the host's LLM
to generate text.

---

### Q4: How do you build an MCP server in Python?

**A:** Using the official `mcp` Python SDK:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()  # Runs with stdio transport by default
```

The SDK provides two APIs:
- **FastMCP** (high-level): Decorator-based, auto-generates schemas from type hints.
- **Server** (low-level): Manual handler registration with `@server.list_tools()` and
  `@server.call_tool()` decorators.

Install with `pip install mcp` (or `pip install "mcp[cli]"` for CLI utilities).

---

### Q5: What transport protocols does MCP support?

**A:** MCP supports three transport mechanisms:

1. **stdio** -- Communication via stdin/stdout. The server runs as a child process. Best
   for local/desktop use. Simple and secure with no network exposure.
2. **Streamable HTTP** -- A single HTTP endpoint (POST) that can optionally upgrade
   responses to SSE streams. Supports session management via `Mcp-Session-Id` headers.
   Best for remote/cloud deployments. This is the current recommended HTTP transport
   (protocol version 2025-03-26).
3. **SSE (legacy)** -- The original HTTP transport using two endpoints (GET /sse for
   events, POST /messages for requests). Deprecated in favor of Streamable HTTP.

All transports use JSON-RPC 2.0 as the message format.

---

### Q6: How do MCP resources differ from tools?

**A:** Key differences:

| Aspect        | Resources                            | Tools                               |
|---------------|--------------------------------------|--------------------------------------|
| Purpose       | Provide data/context to the LLM      | Perform actions with side effects    |
| Control       | Application-controlled               | Model-controlled (LLM decides)       |
| Read/Write    | Read-only                            | Can read and write                   |
| Identification| URI-based (`file://`, `db://`)       | Name-based                           |
| Schema        | MIME type for content format          | JSON Schema for input parameters     |
| Side effects  | None                                 | Can modify state                     |

Resources are analogous to GET requests (safe, idempotent), while tools are analogous to
POST/PUT/DELETE requests (can have side effects).

---

### Q7: What is sampling in MCP?

**A:** Sampling is a feature where the MCP **server** can request the host to perform an
LLM completion. The flow is:

1. During a tool call, the server realizes it needs LLM reasoning.
2. The server sends a `sampling/createMessage` request to the client.
3. The host may show this to the user for approval, modify it, or apply policies.
4. The host runs the LLM and returns the result to the server.
5. The server uses this result to complete its work.

This enables **agentic behavior** within servers -- a server can delegate complex
reasoning to the LLM without the server itself running a model. The host always maintains
control over what the LLM sees and produces (human-in-the-loop).

---

### Q8: How do you handle errors in MCP servers?

**A:** MCP has two levels of error handling:

1. **Tool-level errors** (expected): Return a `TextContent` with `isError=True`. The LLM
   sees this as a tool result and can adjust its approach.

   ```python
   return [TextContent(type="text", text="File not found", isError=True)]
   ```

2. **Protocol-level errors** (unexpected): Raise an `McpError` with a JSON-RPC error
   code. These indicate problems with the protocol itself.

   ```python
   raise McpError(INTERNAL_ERROR, "Database connection lost")
   ```

Best practice: catch expected errors and return them as tool results so the LLM can
recover. Only raise `McpError` for truly unexpected conditions.

---

### Q9: How do you deploy MCP servers in production?

**A:** Common deployment patterns:

1. **Sidecar**: Server runs alongside the application as a child process (stdio). Simple
   but limited to single-host deployments.
2. **Standalone HTTP service**: Deploy the server as a web service using Streamable HTTP
   transport. Can run in containers, Kubernetes, or serverless functions.
3. **Gateway pattern**: A single MCP server that aggregates multiple backend services.

Production considerations:
- Use environment variables or config files for secrets (never hardcode).
- Implement authentication (API keys, OAuth, JWT) for HTTP transport.
- Add rate limiting to prevent abuse.
- Use structured logging for audit trails.
- Monitor server health with metrics.
- Use connection pooling for databases.
- Implement graceful shutdown with lifespan management.

---

### Q10: What security considerations exist for MCP?

**A:** Key security areas:

1. **Input validation**: Sanitize all inputs -- prevent SQL injection, path traversal,
   command injection. Use allowlists rather than denylists.
2. **Authentication**: For HTTP transport, implement token-based auth (Bearer tokens,
   API keys). For stdio, rely on OS-level process isolation.
3. **Authorization**: Enforce per-tool, per-user permissions. Not all users should access
   all tools.
4. **Path safety**: For file-access servers, resolve paths and ensure they stay within
   allowed directories (`Path.is_relative_to()`).
5. **Query safety**: For database servers, use parameterized queries, restrict to SELECT,
   block dangerous keywords.
6. **Rate limiting**: Prevent abuse by limiting tool calls per client per time window.
7. **Audit logging**: Log all tool calls with arguments for security review.
8. **Host mediation**: The host should implement human-in-the-loop approval for
   destructive operations.
9. **Least privilege**: Servers should request only the permissions they need.
10. **Transport security**: Use HTTPS for remote connections, never transmit secrets in
    plain text.

---

### Q11: How does MCP integrate with LangGraph?

**A:** MCP integrates with LangGraph through the `langchain-mcp-adapters` library:

1. `MultiServerMCPClient` connects to multiple MCP servers.
2. It wraps MCP tools as LangChain `BaseTool` instances.
3. These tools are passed to a LangGraph agent (e.g., `create_react_agent`).
4. The agent can then use tools from any connected MCP server transparently.

This means you can write MCP servers once and use them in any LangGraph workflow, or
conversely, use existing MCP servers from the ecosystem in your LangGraph agents without
writing custom tool wrappers.

---

### Q12: What is the difference between stdio and Streamable HTTP transport?

**A:**

| Feature              | stdio                          | Streamable HTTP              |
|----------------------|--------------------------------|------------------------------|
| Connection model     | Server is a child process      | Server is an HTTP service    |
| Network access       | No network, local only         | Can be remote                |
| Multiple clients     | No (1:1 with parent process)   | Yes (HTTP server)            |
| Session management   | Implicit (process lifetime)    | Explicit (Mcp-Session-Id)    |
| Deployment           | Desktop/CLI                    | Cloud, containers, serverless|
| Scalability          | Single user                    | Multiple concurrent clients  |
| Security             | OS process isolation           | Need auth tokens, TLS        |
| Streaming            | N/A (pipe-based)               | Optional SSE upgrade         |

**When to use stdio**: Desktop applications, CLI tools, development, single-user.

**When to use Streamable HTTP**: Cloud services, shared servers, multi-tenant, production.

---

### Q13: How do you test MCP servers?

**A:** Several approaches:

1. **MCP Inspector**: A developer tool provided by the SDK for interactive testing.

   ```bash
   mcp dev my_server.py
   ```

2. **Programmatic testing**: Write a test client that connects and calls tools:

   ```python
   import pytest
   from mcp import ClientSession, StdioServerParameters
   from mcp.client.stdio import stdio_client

   @pytest.mark.asyncio
   async def test_weather_tool():
       params = StdioServerParameters(command="python", args=["server.py"])
       async with stdio_client(params) as (read, write):
           async with ClientSession(read, write) as session:
               await session.initialize()
               result = await session.call_tool(
                   "get_weather", arguments={"city": "london"}
               )
               assert "london" in result.content[0].text.lower()
               assert "Error" not in result.content[0].text
   ```

3. **Unit testing tool handlers**: Test the underlying functions directly without the MCP
   protocol layer.

4. **Integration testing**: Test the full flow including LLM interaction with mock or
   real models.

---

### Q14: What are resource subscriptions?

**A:** Resource subscriptions allow clients to be notified when a resource's content
changes. The flow:

1. Client sends `resources/subscribe` with a resource URI.
2. Server tracks the subscription.
3. When the resource content changes, the server sends a
   `notifications/resources/updated` notification.
4. The client can then re-read the resource to get the updated content.
5. Client can send `resources/unsubscribe` to stop receiving notifications.

This is useful for live data (monitoring dashboards, log files, config changes) where
the client needs to stay up-to-date without polling.

---

### Q15: How does MCP solve the M x N integration problem?

**A:** Without MCP, if you have M AI applications (Claude Desktop, VS Code Copilot,
custom apps) and N data sources/tools (GitHub, Slack, databases), each app needs a
custom integration for each tool = M x N total integrations.

With MCP: each application implements the MCP client protocol once (M clients), and each
tool implements the MCP server protocol once (N servers). Any client can connect to any
server through the standardized protocol = M + N total implementations.

Example: 5 apps x 10 tools = 50 custom integrations vs. 5 + 10 = 15 MCP implementations.

---

### Q16: Can MCP servers call other MCP servers?

**A:** Yes. An MCP server can also act as an MCP client, connecting to other MCP servers.
This enables **composition** and **gateway** patterns. For example, an "orchestrator"
server could connect to a database server and a search server, combining their
capabilities into higher-level tools. However, each connection is still a 1:1
client-server relationship.

---

### Q17: What is the message format used by MCP?

**A:** MCP uses **JSON-RPC 2.0** as its wire format. Every message is a JSON object with:

- **Requests**: `{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {...}}`
- **Responses**: `{"jsonrpc": "2.0", "id": 1, "result": {...}}`
- **Errors**: `{"jsonrpc": "2.0", "id": 1, "error": {"code": -32600, "message": "..."}}`
- **Notifications**: `{"jsonrpc": "2.0", "method": "notifications/...", "params": {...}}`
  (no `id` field -- no response expected)

---

### Q18: How does the initialization handshake work?

**A:** The initialization follows three steps:

1. **Client sends `initialize`**: Includes protocol version, client capabilities, and
   client info.
2. **Server responds**: Returns its protocol version, server capabilities (which
   primitives it supports), and server info.
3. **Client sends `initialized` notification**: Confirms the handshake is complete.

During initialization, both sides negotiate:
- Protocol version (must be compatible).
- Capabilities (which features each side supports: tools, resources, prompts, sampling,
  subscriptions, etc.).

No other requests are allowed until initialization is complete.

---

### Q19: What is FastMCP and how does it differ from the low-level Server API?

**A:** FastMCP is a high-level, decorator-based API in the `mcp` Python SDK that reduces
boilerplate:

| Feature             | FastMCP                          | Low-Level Server               |
|---------------------|----------------------------------|---------------------------------|
| Tool definition     | `@mcp.tool()` on any function    | `@server.list_tools()` + `@server.call_tool()` |
| Schema generation   | Auto from type hints + docstring | Manual JSON Schema              |
| Resource definition | `@mcp.resource("uri")`           | `@server.list_resources()` + `@server.read_resource()` |
| Validation          | Automatic via type hints         | Manual                          |
| Lines of code       | ~5 per tool                      | ~20-30 per tool                 |

FastMCP is recommended for most use cases. The low-level API is useful when you need full
control over the protocol or have complex routing logic.

---

### Q20: How do you handle long-running operations in MCP?

**A:** MCP provides several mechanisms:

1. **Progress reporting**: The server sends progress notifications with a token, current
   progress, total, and optional message. The client can display a progress indicator.
2. **Cancellation**: The client can send a `notifications/cancelled` message with the
   request ID. The server should check for cancellation and abort gracefully.
3. **Timeouts**: Both client and server can implement timeouts. The transport layer also
   has connection timeouts.
4. **Async execution**: Use `asyncio` to run operations concurrently without blocking
   the server.

```python
@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "long_task":
        ctx = server.request_context
        for i in range(100):
            # Report progress
            if ctx.meta and ctx.meta.progressToken:
                await ctx.session.send_progress_notification(
                    progress_token=ctx.meta.progressToken,
                    progress=i + 1,
                    total=100,
                )
            await asyncio.sleep(0.1)  # Simulate work
        return [TextContent(type="text", text="Task complete")]
```

---

### Q21: What are tool annotations and why are they useful?

**A:** Tool annotations are optional metadata hints that describe a tool's behavior:

- `readOnlyHint` -- Whether the tool only reads data.
- `destructiveHint` -- Whether the tool can destroy/delete data.
- `idempotentHint` -- Whether calling it multiple times has the same effect.
- `openWorldHint` -- Whether the tool interacts with external systems.

These help the **host** make informed decisions about confirmation policies. For example,
a host might auto-approve `readOnlyHint=True` tools but require explicit user
confirmation for `destructiveHint=True` tools. They are hints, not guarantees -- the host
ultimately decides how to use them.

---

### Q22: How does MCP compare to OpenAI's function calling?

**A:**

| Aspect              | MCP                                    | OpenAI Function Calling           |
|---------------------|----------------------------------------|-----------------------------------|
| Scope               | Full protocol (tools + resources + prompts) | Tools only                    |
| Standardization     | Open standard, vendor-agnostic         | OpenAI-specific API               |
| Architecture        | Client-server with transport layer     | Part of chat completions API      |
| Discovery           | Runtime tool/resource discovery         | Tools defined at call time        |
| Execution           | Server executes tools                  | Application executes functions    |
| Data access         | Resources for context                  | No equivalent                     |
| Multi-provider      | Any LLM can use MCP                    | OpenAI models only                |
| Ecosystem           | Shared servers across apps             | Per-application tool definitions  |

MCP is a broader standard that includes function calling as one capability (tools), but
also adds resources, prompts, sampling, and a full client-server architecture.

---

### Q23: Can you use MCP with models other than Claude?

**A:** Yes. MCP is an open protocol and is model-agnostic. Any application that
implements the MCP client protocol can connect to any MCP server, regardless of which LLM
it uses. The `langchain-mcp-adapters` library, for example, allows MCP tools to be used
with any LangChain-supported model (OpenAI, Anthropic, Google, open-source models, etc.).

The protocol itself has no dependency on Claude or Anthropic -- it is a pure
communication standard.

---

### Q24: What happens if an MCP server crashes during a tool call?

**A:** When a server crashes:

1. **stdio transport**: The server process exits, closing stdin/stdout. The client
   detects the broken pipe and receives an error for the pending request. The host can
   choose to restart the server and retry.
2. **Streamable HTTP**: The HTTP connection drops. The client receives a connection error.
   If the server supports session resumption (via `Mcp-Session-Id`), the client may
   reconnect to the same session. Otherwise, a new session is needed.

Best practices for resilience:
- Implement automatic server restart in the host.
- Use health checks for HTTP servers.
- Make tool operations idempotent where possible.
- Log all errors for debugging.

---

### Q25: How do you version MCP servers?

**A:** Versioning strategies:

1. **Protocol version**: The MCP protocol itself is versioned (e.g., "2025-03-26").
   Client and server negotiate during initialization. The protocol is designed for
   backward compatibility.
2. **Server version**: Declared in `serverInfo` during initialization (name + version).
   Clients can inspect this.
3. **Tool versioning**: Avoid breaking changes to existing tools. Instead, add new tools
   with different names (e.g., `query_v2`) or add optional parameters with defaults.
4. **Capability negotiation**: During initialization, both sides declare what features
   they support. This allows gradual adoption of new features.

---

### Bonus Q: What are the key things to remember for an MCP interview?

**A:** The top points to emphasize:

1. MCP solves the **M x N integration problem** by standardizing LLM-tool communication.
2. Architecture has three roles: **Host, Client, Server** -- clients have 1:1
   relationships with servers.
3. Four primitives: **Resources** (read-only context), **Tools** (actions),
   **Prompts** (templates), **Sampling** (server-initiated LLM calls).
4. Communication uses **JSON-RPC 2.0** over **stdio** or **Streamable HTTP**.
5. The Python SDK offers both **low-level** (`Server`) and **high-level** (`FastMCP`)
   APIs.
6. Security is critical: **input validation, authentication, authorization, rate
   limiting, audit logging**.
7. MCP is **model-agnostic** -- works with any LLM, not just Claude.
8. The **ecosystem** is growing -- many pre-built servers are available.
9. In production, use **lifespan management**, **connection pooling**, **structured
   logging**, and **health checks**.
10. MCP enables **composability** -- servers can chain together for complex workflows.

---

## Summary Cheat Sheet

```
MCP AT A GLANCE
================

Protocol:     JSON-RPC 2.0
Transport:    stdio | Streamable HTTP | SSE (legacy)
Architecture: Host -> Client(s) -> Server(s)

Primitives:
  Resources  = Data for context      (application-controlled)
  Tools      = Functions with effects (model-controlled)
  Prompts    = Reusable templates    (user-controlled)
  Sampling   = Server asks LLM      (host-mediated)

Python SDK:
  pip install mcp
  FastMCP  = High-level decorators  (@mcp.tool, @mcp.resource)
  Server   = Low-level handlers     (@server.list_tools, @server.call_tool)

Lifecycle:
  initialize -> initialized -> (normal operation) -> shutdown

Security:
  Validate inputs | Authenticate | Authorize | Rate-limit | Audit

Key Benefit:
  M x N integrations  -->  M + N integrations
```
