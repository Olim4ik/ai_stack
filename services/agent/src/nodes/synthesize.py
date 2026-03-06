"""Synthesize node — generate final answer from gathered context."""

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from ..state import AgentState

logger = structlog.get_logger()

SYNTHESIZE_PROMPT = """You are a helpful internal DevOps assistant. Answer the user's question based on the provided context.

Rules:
- Only use information from the provided context
- Cite sources by referencing the document title or source file
- If the context doesn't contain enough information, say so
- Be concise and actionable
- Format your response with markdown when helpful

## Retrieved Context
{context}

## Tool Results
{tool_results}

## User Question
{query}"""


async def synthesize_node(state: AgentState, *, llm: BaseChatModel) -> dict:
    """Generate a final cited answer from all gathered context."""
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    tool_results = state.get("tool_results", [])

    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks):
        source = chunk.get("metadata", {}).get("source_file", "unknown")
        section = chunk.get("metadata", {}).get("section", "")
        label = f"[Source: {source}"
        if section:
            label += f" > {section}"
        label += f", score: {chunk.get('score', 0):.2f}]"
        context_parts.append(f"{label}\n{chunk.get('text', '')}")

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No documents found."

    # Build tool results summary
    tool_summary_parts = []
    for tr in tool_results:
        tool_summary_parts.append(f"Tool: {tr.get('tool', 'unknown')}\nResult: {tr.get('result', tr.get('error', 'N/A'))}")
    tool_summary = "\n\n".join(tool_summary_parts) if tool_summary_parts else "No tool results."

    response = await llm.ainvoke([
        HumanMessage(content=SYNTHESIZE_PROMPT.format(
            context=context,
            tool_results=tool_summary,
            query=query,
        )),
    ])

    answer = response.content
    logger.info("Answer synthesized", length=len(answer))

    return {
        "final_answer": answer,
        "reasoning_trace": state.get("reasoning_trace", []) + [
            {"node": "synthesize", "action": "Generated final answer"}
        ],
    }
