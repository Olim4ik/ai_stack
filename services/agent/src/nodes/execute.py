"""Execute node — run plan steps by calling retrieval or MCP tools."""

import structlog

from ..grpc_clients.retrieval import RetrievalClient
from ..mcp_client.client import MCPClientWrapper
from ..state import AgentState

logger = structlog.get_logger()

CONFIRMATION_TOOLS = {"jira_create", "pagerduty_ack"}


async def execute_node(
    state: AgentState,
    *,
    retrieval_client: RetrievalClient,
    mcp_client: MCPClientWrapper,
) -> dict:
    """Execute plan steps — search knowledge base or call MCP tools."""
    plan = state.get("plan", [])
    team = state.get("team", "platform")
    collection = f"team_{team}"

    retrieved_chunks = list(state.get("retrieved_chunks", []))
    tool_results = list(state.get("tool_results", []))
    trace = list(state.get("reasoning_trace", []))
    pending_confirmation = None

    for step in plan:
        if step.startswith("SEARCH:"):
            search_query = step[7:].strip()
            chunks = await retrieval_client.search(query=search_query, collection=collection, top_k=3)
            retrieved_chunks.extend(chunks)
            trace.append({"node": "execute", "action": f"Searched: '{search_query}', got {len(chunks)} results"})
            logger.info("Execute: search", query=search_query, results=len(chunks))

        elif step.startswith("TOOL:"):
            parts = step[5:].strip().split(":", 1)
            tool_name = parts[0].strip()
            tool_desc = parts[1].strip() if len(parts) > 1 else ""

            # Check if this tool requires confirmation
            if tool_name in CONFIRMATION_TOOLS:
                pending_confirmation = {
                    "tool": tool_name,
                    "description": tool_desc,
                    "step": step,
                }
                trace.append({"node": "execute", "action": f"Confirmation required for {tool_name}: {tool_desc}"})
                logger.info("Execute: confirmation required", tool=tool_name)
                break  # Stop execution, wait for confirmation

            try:
                result = await mcp_client.call_tool(tool_name, {})
                tool_results.append({"tool": tool_name, "result": result})
                trace.append({"node": "execute", "action": f"Called tool {tool_name}", "result_preview": str(result)[:200]})
                logger.info("Execute: tool called", tool=tool_name)
            except Exception as e:
                tool_results.append({"tool": tool_name, "error": str(e)})
                trace.append({"node": "execute", "action": f"Tool {tool_name} failed: {e}"})
                logger.error("Execute: tool failed", tool=tool_name, error=str(e))

    return {
        "retrieved_chunks": retrieved_chunks,
        "tool_results": tool_results,
        "pending_confirmation": pending_confirmation,
        "reasoning_trace": trace,
    }
