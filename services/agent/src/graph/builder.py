"""Build and compile the LangGraph agent graph."""

from functools import partial

import structlog
from anthropic import AsyncAnthropic
from langgraph.graph import END, StateGraph

from ..grpc_clients.retrieval import RetrievalClient
from ..mcp_client.client import MCPClientWrapper
from ..nodes.classify import classify_node
from ..nodes.confirm import confirm_node
from ..nodes.execute import execute_node
from ..nodes.plan import plan_node
from ..nodes.retrieve import retrieve_node
from ..nodes.synthesize import synthesize_node
from ..state import AgentState
from .router import route_after_execute, route_by_query_type

logger = structlog.get_logger()


def build_graph(
    anthropic_client: AsyncAnthropic,
    model: str,
    retrieval_client: RetrievalClient,
    mcp_client: MCPClientWrapper,
):
    """Build the LangGraph state graph.

    Graph structure:
        START → classify → {simple: retrieve, multi_step: plan, action: execute}
        retrieve → synthesize → END
        plan → execute → {needs_confirm: confirm, else: synthesize} → END
        confirm → synthesize → END
    """
    graph = StateGraph(AgentState)

    # Bind dependencies to nodes via partial
    _classify = partial(classify_node, anthropic_client=anthropic_client, model=model)
    _retrieve = partial(retrieve_node, retrieval_client=retrieval_client)
    _plan = partial(plan_node, anthropic_client=anthropic_client, model=model)
    _execute = partial(execute_node, retrieval_client=retrieval_client, mcp_client=mcp_client)
    _synthesize = partial(synthesize_node, anthropic_client=anthropic_client, model=model)

    # Add nodes
    graph.add_node("classify", _classify)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("plan", _plan)
    graph.add_node("execute", _execute)
    graph.add_node("confirm", _confirm_passthrough)
    graph.add_node("synthesize", _synthesize)

    # Set entry point
    graph.set_entry_point("classify")

    # Conditional routing after classify
    graph.add_conditional_edges("classify", route_by_query_type, {
        "retrieve": "retrieve",
        "plan": "plan",
        "execute": "execute",
    })

    # Simple path: retrieve → synthesize → END
    graph.add_edge("retrieve", "synthesize")

    # Multi-step path: plan → execute → (confirm or synthesize) → END
    graph.add_edge("plan", "execute")
    graph.add_conditional_edges("execute", route_after_execute, {
        "confirm": "confirm",
        "synthesize": "synthesize",
    })

    # Confirm → synthesize → END
    graph.add_edge("confirm", "synthesize")
    graph.add_edge("synthesize", END)

    # Compile with interrupt_before on confirm for human-in-the-loop
    compiled = graph.compile(interrupt_before=["confirm"])

    logger.info("LangGraph agent graph compiled")
    return compiled


async def _confirm_passthrough(state: AgentState) -> dict:
    """Passthrough for confirm node — actual logic in confirm.py.

    When the graph resumes after interrupt, this node simply passes through.
    """
    return await confirm_node(state)
