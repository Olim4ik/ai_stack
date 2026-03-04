"""Plan node — decompose a multi-step query into sub-tasks."""

import json

import structlog
from anthropic import AsyncAnthropic

from ..state import AgentState

logger = structlog.get_logger()

PLAN_PROMPT = """You are a planning assistant. Decompose the following complex query into 2-5 sequential sub-tasks that can be executed one at a time.

Each sub-task should be either:
- A knowledge base search (prefix with "SEARCH: ")
- A tool action (prefix with "TOOL: tool_name: description")

Respond with ONLY a JSON array of strings.

Example:
["SEARCH: Find auth service restart procedure", "SEARCH: Find auth service health check endpoints", "TOOL: github_get_file: Check the auth service docker-compose.yml"]

User query: {query}"""


async def plan_node(state: AgentState, *, anthropic_client: AsyncAnthropic, model: str) -> dict:
    """Decompose a multi-step query into sub-tasks."""
    query = state["query"]

    response = await anthropic_client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": PLAN_PROMPT.format(query=query)}],
    )

    text = response.content[0].text.strip()

    try:
        plan = json.loads(text)
        if not isinstance(plan, list):
            plan = [str(plan)]
    except json.JSONDecodeError:
        # Fallback: single search step
        plan = [f"SEARCH: {query}"]

    logger.info("Plan created", steps=len(plan))

    return {
        "plan": plan,
        "reasoning_trace": state.get("reasoning_trace", []) + [
            {"node": "plan", "action": f"Created plan with {len(plan)} steps", "steps": plan}
        ],
    }
