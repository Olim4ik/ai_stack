"""Classify node — categorize the user query."""

import json

import structlog
from anthropic import AsyncAnthropic

from ..state import AgentState

logger = structlog.get_logger()

CLASSIFY_PROMPT = """Classify the following user query into one of these categories:

1. "simple" — A straightforward question that can be answered by searching the knowledge base.
   Examples: "How do I restart the auth service?", "What is our deployment process?"

2. "multi_step" — A complex question that requires multiple searches or reasoning steps.
   Examples: "Compare our auth service architecture with the payment service", "Find all runbooks related to database migrations and summarize the common patterns"

3. "action" — A request to perform an action using external tools (Jira, GitHub, PagerDuty).
   Examples: "Create a Jira ticket for the auth bug", "Check if there are any open PagerDuty incidents", "What's in PR #123?"

Respond with ONLY a JSON object: {"query_type": "simple" | "multi_step" | "action"}

User query: {query}"""


async def classify_node(state: AgentState, *, anthropic_client: AsyncAnthropic, model: str) -> dict:
    """Classify the user query into simple/multi_step/action."""
    query = state["query"]

    response = await anthropic_client.messages.create(
        model=model,
        max_tokens=100,
        messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(query=query)}],
    )

    text = response.content[0].text.strip()

    try:
        result = json.loads(text)
        query_type = result.get("query_type", "simple")
    except json.JSONDecodeError:
        # Fallback: try to extract from text
        if "multi_step" in text:
            query_type = "multi_step"
        elif "action" in text:
            query_type = "action"
        else:
            query_type = "simple"

    logger.info("Query classified", query=query[:50], query_type=query_type)

    return {
        "query_type": query_type,
        "reasoning_trace": state.get("reasoning_trace", []) + [
            {"node": "classify", "action": f"Classified query as '{query_type}'"}
        ],
    }
