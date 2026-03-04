# 05. LangGraph: Stateful Agent Workflows

> Interview preparation document for Backend AI Engineers.
> Covers theory, architecture diagrams, production code examples, and Q&A.
> Based on **LangGraph 0.2+** (the `langgraph` package, part of the LangChain ecosystem).

---

## Table of Contents

1. [LangGraph Overview](#1-langgraph-overview)
2. [Core Concepts](#2-core-concepts)
3. [Building a Basic Graph](#3-building-a-basic-graph)
4. [Conditional Routing](#4-conditional-routing)
5. [Checkpointing and Persistence](#5-checkpointing-and-persistence)
6. [Human-in-the-Loop](#6-human-in-the-loop)
7. [Subgraphs](#7-subgraphs)
8. [Multi-Agent Patterns](#8-multi-agent-patterns)
9. [Tool Integration](#9-tool-integration)
10. [State Management Patterns](#10-state-management-patterns)
11. [Production Patterns](#11-production-patterns)
12. [Practical Examples](#12-practical-examples)
13. [Q&A Section](#13-qa-section)

---

## 1. LangGraph Overview

### 1.1 What Is LangGraph?

LangGraph is a **framework for building stateful, multi-step agent applications** as directed graphs. It is part of the broader LangChain ecosystem but lives in its own package (`langgraph`). While LangChain provides composable chains and tool abstractions, LangGraph adds explicit **graph-based control flow** on top -- letting you define nodes, edges, conditional branching, loops, persistence, and human-in-the-loop interactions with full visibility into every step.

Key facts:

| Attribute | Detail |
|-----------|--------|
| Package | `langgraph` (PyPI) |
| Maintained by | LangChain Inc. |
| Core abstraction | `StateGraph` -- a directed graph where nodes transform shared state |
| Language support | Python (primary), JavaScript/TypeScript |
| LLM agnostic | Works with any LLM via LangChain chat model interface |
| Production server | LangGraph Platform / LangGraph Cloud for deployment |

### 1.2 Why LangGraph Over Plain LangChain Agents?

Plain LangChain agents (`AgentExecutor`) run in a simple loop: call LLM, maybe call a tool, repeat. This works for toy demos but breaks down in production:

| Limitation of AgentExecutor | LangGraph Solution |
|-----------------------------|--------------------|
| Opaque loop -- hard to debug or customize | Explicit graph with named nodes you can inspect |
| No built-in persistence | Checkpointers (memory, SQLite, Postgres) |
| No human-in-the-loop | `interrupt_before` / `interrupt_after` |
| No multi-agent coordination | Subgraphs, supervisor pattern, shared state |
| Poor error handling | Per-node retry, fallback edges, error routing |
| Cannot resume after failure | Checkpoint + replay from any saved state |
| Cannot stream intermediate steps cleanly | First-class streaming of events, tokens, state deltas |

> **Interview tip**: LangChain's own documentation now recommends LangGraph as the successor to `AgentExecutor` for any non-trivial agent workflow.

### 1.3 Comparison with Other Orchestration Frameworks

| Feature | LangGraph | CrewAI | AutoGen | DSPy |
|---------|-----------|--------|---------|------|
| Paradigm | Graph-based | Role-based agents | Conversational agents | Compiler-optimized prompts |
| Control flow | Explicit (nodes + edges) | Implicit (task delegation) | Message passing | Pipeline |
| State management | TypedDict / Pydantic with reducers | Internal per-agent memory | Chat history | Signatures |
| Persistence | Built-in checkpointers | External | External | None |
| Human-in-the-loop | First-class (`interrupt`) | Limited | Limited | None |
| Streaming | First-class | Limited | Limited | None |
| Debugging | Graph visualization, LangSmith | Logs | Logs | Traces |
| Multi-agent | Subgraphs, supervisor | Crews with roles | Group chat | N/A |
| Production readiness | High (LangGraph Platform) | Medium | Medium | Medium |
| Learning curve | Medium | Low | Low | High |

### 1.4 When to Use LangGraph

**Use LangGraph when you need:**
- Explicit, inspectable control flow over an agent's decisions
- Persistence, checkpointing, time travel
- Human-in-the-loop approval steps
- Multi-agent coordination (supervisor, hierarchical teams)
- Streaming of intermediate results in production
- Complex branching, loops, or parallel execution paths

**Do NOT need LangGraph when:**
- A single LLM call with a prompt template suffices
- A linear chain (prompt -> LLM -> parser) is all you need
- You are prototyping and do not need persistence or debugging

---

## 2. Core Concepts

### 2.1 Conceptual Architecture

```
         ┌──────────────────────────────────────────────────────────┐
         │                      StateGraph                         │
         │                                                          │
         │   ┌──────────┐                                          │
         │   │  START    │  (virtual entry point)                   │
         │   └────┬─────┘                                          │
         │        │                                                 │
         │   ┌────▼─────┐       normal edge                        │
         │   │  Node A   │──────────────────┐                      │
         │   │ (function)│                  │                      │
         │   └────┬─────┘                  │                      │
         │        │                         │                      │
         │   ┌────▼──────────┐        ┌────▼─────┐                │
         │   │  Conditional   │        │  Node C   │                │
         │   │  Edge (router) │        │ (function)│                │
         │   └──┬────────┬───┘        └────┬─────┘                │
         │      │        │                  │                      │
         │ ┌────▼───┐ ┌──▼────┐            │                      │
         │ │ Node B  │ │ Node D │            │                      │
         │ └────┬───┘ └──┬────┘            │                      │
         │      │        │                  │                      │
         │      └────┬───┘──────────────────┘                      │
         │           │                                              │
         │      ┌────▼─────┐                                       │
         │      │   END     │  (virtual exit point)                 │
         │      └──────────┘                                       │
         │                                                          │
         └──────────────────────────────────────────────────────────┘
```

### 2.2 State

State is a **typed data structure** (typically a `TypedDict` or Pydantic `BaseModel`) that flows through every node in the graph. Each node reads the current state, performs work, and returns a **partial update** that gets merged back.

```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from operator import add


class AgentState(TypedDict):
    """The state that flows through the entire graph."""
    messages: Annotated[list[BaseMessage], add]   # reducer: append
    current_step: str
    final_answer: str | None
```

Key rules:
- Nodes receive the **full** state and return a **partial** dict.
- The framework merges the partial update into the current state.
- Fields with `Annotated[..., reducer]` use the reducer function instead of replacement (see Section 10).

### 2.3 Nodes

A node is any **callable** that accepts the state and returns a partial state update. It can be a plain function, a class method, or a runnable.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

def call_agent(state: AgentState) -> dict:
    """Node: invoke the LLM with the current messages."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

### 2.4 Edges

Edges connect nodes and define execution order.

| Edge Type | Description | API |
|-----------|-------------|-----|
| Normal edge | Always follow this path | `graph.add_edge("A", "B")` |
| Conditional edge | Choose path based on state | `graph.add_conditional_edges("A", router_fn, mapping)` |
| Entry edge | From START to the first node | `graph.add_edge(START, "A")` |
| Finish edge | From a node to END | `graph.add_edge("A", END)` |

### 2.5 StateGraph

`StateGraph` is the central class. You construct it with a state schema, add nodes and edges, then **compile** it into a runnable.

```python
from langgraph.graph import StateGraph, START, END

graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", call_agent)
graph_builder.add_edge(START, "agent")
graph_builder.add_edge("agent", END)

graph = graph_builder.compile()
```

### 2.6 START and END

- `START` -- a virtual node representing the entry point. You must connect it to at least one real node.
- `END` -- a virtual node representing termination. When execution reaches `END`, the graph returns the final state.

These are imported from `langgraph.graph`:
```python
from langgraph.graph import START, END
```

---

## 3. Building a Basic Graph

### 3.1 Step-by-Step Walkthrough

Below is a complete example of a simple agent that can answer questions using tools.

**Step 1 -- Define the State**

```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from operator import add


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add]
```

**Step 2 -- Define Node Functions**

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def chatbot(state: AgentState) -> dict:
    """Call the LLM and return its response."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

**Step 3 -- Build the Graph**

```python
from langgraph.graph import StateGraph, START, END

# 1. Create graph builder with state schema
builder = StateGraph(AgentState)

# 2. Add nodes
builder.add_node("chatbot", chatbot)

# 3. Add edges
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# 4. Compile
graph = builder.compile()
```

**Step 4 -- Run the Graph**

```python
from langchain_core.messages import HumanMessage

# Invoke returns the final state
result = graph.invoke({
    "messages": [HumanMessage(content="What is LangGraph?")]
})

# The last message is the AI response
print(result["messages"][-1].content)
```

**Step 5 -- Visualize (optional)**

```python
# Print ASCII representation
graph.get_graph().print_ascii()

# Or generate a Mermaid diagram
print(graph.get_graph().draw_mermaid())

# Or save as PNG (requires graphviz)
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
```

### 3.2 Graph Execution Flow Diagram

```
invoke({"messages": [HumanMessage("What is LangGraph?")]})
    │
    │  Initial state: {messages: [HumanMessage(...)]}
    │
    ▼
┌──────────┐
│  START    │
└────┬─────┘
     │
     │  state passed to node
     ▼
┌──────────┐
│ chatbot   │  --> calls llm.invoke(state["messages"])
│           │  --> returns {"messages": [AIMessage(...)]}
└────┬─────┘
     │
     │  state merged: {messages: [HumanMessage(...), AIMessage(...)]}
     ▼
┌──────────┐
│   END     │
└──────────┘
     │
     ▼
returns final state: {messages: [HumanMessage(...), AIMessage(...)]}
```

---

## 4. Conditional Routing

Conditional routing is the mechanism that makes LangGraph agents **agentic** -- the graph can branch based on the LLM's output (e.g., whether it wants to call a tool or give a final answer).

### 4.1 add_conditional_edges

```python
def should_continue(state: AgentState) -> str:
    """Router: decide the next node based on the last message."""
    last_message = state["messages"][-1]
    # If the LLM returned tool calls, go to the tool node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, we are done
    return "end"


builder.add_conditional_edges(
    "agent",                # source node
    should_continue,        # router function
    {                       # mapping: router return value -> node name
        "tools": "tool_node",
        "end": END,
    },
)
```

The mapping dict is optional. If omitted, the router function's return value is used directly as the node name (it must exactly match a node name or be `END`).

### 4.2 Multi-Path Branching

A router can return more than two values:

```python
def route_by_intent(state: AgentState) -> str:
    intent = state.get("intent", "unknown")
    if intent == "search":
        return "search_node"
    elif intent == "calculate":
        return "calc_node"
    elif intent == "summarize":
        return "summary_node"
    else:
        return "fallback_node"


builder.add_conditional_edges("classifier", route_by_intent, {
    "search_node": "search_node",
    "calc_node": "calc_node",
    "summary_node": "summary_node",
    "fallback_node": "fallback_node",
})
```

Diagram:

```
                ┌───────────────┐
                │   classifier   │
                └───────┬───────┘
                        │
              ┌─────────┼─────────┬────────────┐
              │         │         │            │
         ┌────▼──┐ ┌────▼──┐ ┌───▼────┐ ┌────▼─────┐
         │search │ │ calc  │ │summary │ │ fallback │
         └───┬───┘ └───┬───┘ └───┬────┘ └────┬─────┘
             │         │         │            │
             └─────────┴─────────┴────────────┘
                        │
                   ┌────▼───┐
                   │  END    │
                   └────────┘
```

### 4.3 Loops (ReAct Pattern)

The ReAct pattern is the canonical loop in LangGraph: the agent calls an LLM, the LLM may request tool use, the tool executes and feeds results back to the LLM, and this repeats until the LLM gives a final answer.

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(AgentState)
builder.add_node("agent", call_agent)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "end": END,
})
builder.add_edge("tools", "agent")  # loop back after tool execution

graph = builder.compile()
```

Loop diagram:

```
    ┌──────────┐
    │  START    │
    └────┬─────┘
         │
    ┌────▼─────┐
    │   agent   │◄────────────┐
    └────┬─────┘              │
         │                    │
    ┌────▼──────────┐    ┌────┴─────┐
    │ should_continue│───►│  tools   │
    │  "tools"?      │    └──────────┘
    └────┬──────────┘
         │ "end"
    ┌────▼─────┐
    │   END     │
    └──────────┘
```

### 4.4 Loop Detection and Safeguards

LangGraph provides a `recursion_limit` parameter to prevent infinite loops:

```python
# Default recursion_limit is 25
result = graph.invoke(
    {"messages": [HumanMessage(content="...")]},
    config={"recursion_limit": 50},
)
```

If the limit is reached, a `GraphRecursionError` is raised. You should handle this gracefully:

```python
from langgraph.errors import GraphRecursionError

try:
    result = graph.invoke(inputs, config={"recursion_limit": 30})
except GraphRecursionError:
    # Graceful degradation
    result = {"messages": [AIMessage(content="I could not resolve this in time.")]}
```

---

## 5. Checkpointing and Persistence

Checkpointing is one of LangGraph's most powerful features. After every node execution (called a "superstep"), the framework saves a **snapshot** of the full state. This enables time travel, fault tolerance, and human-in-the-loop workflows.

### 5.1 Checkpointer Backends

| Checkpointer | Use Case | Package |
|--------------|----------|---------|
| `MemorySaver` | Development, testing | `langgraph.checkpoint.memory` |
| `SqliteSaver` | Local persistence, small-scale | `langgraph-checkpoint-sqlite` |
| `PostgresSaver` | Production, multi-instance | `langgraph-checkpoint-postgres` |

### 5.2 MemorySaver (In-Memory)

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Every invocation must include a thread_id in the config
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke(
    {"messages": [HumanMessage(content="Hi, my name is Alice")]},
    config=config,
)

# Continue the same conversation
result = graph.invoke(
    {"messages": [HumanMessage(content="What is my name?")]},
    config=config,
)
# The agent remembers: "Your name is Alice"
```

### 5.3 SqliteSaver

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# File-based SQLite
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    result = graph.invoke(inputs, config={"configurable": {"thread_id": "t1"}})
```

### 5.4 PostgresSaver (Production)

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:pass@localhost:5432/langgraph"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # Create tables on first run
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
    result = graph.invoke(inputs, config={"configurable": {"thread_id": "t1"}})
```

For async usage:

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    await checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
    result = await graph.ainvoke(inputs, config={"configurable": {"thread_id": "t1"}})
```

### 5.5 What Checkpointing Enables

**Time Travel (Replay)**

```python
# Get all states for a thread
states = list(graph.get_state_history(config))

# Each state has a checkpoint_id and the full state
for state in states:
    print(state.config, state.values["messages"][-1].content)

# Replay from a specific checkpoint
old_config = states[2].config  # go back to the 3rd checkpoint
result = graph.invoke(
    {"messages": [HumanMessage(content="Try a different approach")]},
    config=old_config,
)
```

**Fault Tolerance (Resume After Crash)**

If a node raises an exception, the state from the *previous* checkpoint is preserved. You can fix the issue and re-invoke with the same `thread_id` -- execution resumes from the last successful checkpoint.

**Thread Management**

Each `thread_id` represents an independent conversation/session. Multiple threads can run concurrently. The checkpointer handles isolation.

```python
# Thread 1 -- user Alice
graph.invoke(inputs, config={"configurable": {"thread_id": "alice-001"}})

# Thread 2 -- user Bob (completely independent state)
graph.invoke(inputs, config={"configurable": {"thread_id": "bob-001"}})
```

---

## 6. Human-in-the-Loop

Human-in-the-loop (HITL) lets you **pause** graph execution at a specific node, wait for human input or approval, then **resume**.

### 6.1 Interrupt Flow

```
┌───────┐     ┌───────┐     ┌──── PAUSE ────┐     ┌───────┐     ┌───────┐
│ START │────►│ agent │────►│ ⏸ INTERRUPT   │────►│ tools │────►│  END  │
└───────┘     └───────┘     │  (human sees   │     └───────┘     └───────┘
                            │   tool call,   │
                            │   approves or  │
                            │   rejects)     │
                            └────────────────┘
```

### 6.2 interrupt_before

Pause **before** a node executes:

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"],  # pause before the tools node runs
)

config = {"configurable": {"thread_id": "hitl-1"}}

# First invocation -- runs agent, then pauses before tools
result = graph.invoke(
    {"messages": [HumanMessage(content="Delete all files in /tmp")]},
    config=config,
)

# Inspect what tool is about to be called
state = graph.get_state(config)
print(state.values["messages"][-1].tool_calls)
# [{'name': 'delete_files', 'args': {'path': '/tmp'}, 'id': '...'}]

# Human reviews and decides to approve -- just resume
result = graph.invoke(None, config=config)  # pass None to continue

# Or human rejects -- modify state and resume
graph.update_state(config, {
    "messages": [AIMessage(content="I cannot delete files for safety reasons.")]
})
result = graph.invoke(None, config=config)
```

### 6.3 interrupt_after

Pause **after** a node executes (useful to review results before proceeding):

```python
graph = builder.compile(
    checkpointer=memory,
    interrupt_after=["agent"],  # pause after agent runs, before routing
)
```

### 6.4 Manual State Modification

`update_state` lets you inject or modify state at the paused checkpoint:

```python
from langchain_core.messages import HumanMessage, ToolMessage

# Simulate a tool result without actually calling the tool
graph.update_state(
    config,
    {
        "messages": [
            ToolMessage(
                content="Operation cancelled by admin.",
                tool_call_id=state.values["messages"][-1].tool_calls[0]["id"],
            )
        ]
    },
    as_node="tools",  # pretend this update came from the "tools" node
)

# Resume -- the agent will see the injected tool result
result = graph.invoke(None, config=config)
```

### 6.5 Dynamic Interrupt with `interrupt()` Function

LangGraph 0.2.57+ introduced the `interrupt()` function for more flexible HITL:

```python
from langgraph.types import interrupt, Command

def human_review_node(state: AgentState) -> Command:
    """Node that pauses and waits for human input."""
    tool_call = state["messages"][-1].tool_calls[0]

    # This suspends execution and sends data to the client
    human_response = interrupt({
        "question": f"Approve tool call: {tool_call['name']}({tool_call['args']})?",
        "tool_call": tool_call,
    })

    if human_response["approved"]:
        # Continue to tools
        return Command(goto="tools")
    else:
        # Skip tools, inject rejection message
        return Command(
            goto="agent",
            update={
                "messages": [
                    ToolMessage(
                        content=f"Rejected by human: {human_response.get('reason', '')}",
                        tool_call_id=tool_call["id"],
                    )
                ]
            },
        )
```

Resume with:

```python
result = graph.invoke(
    Command(resume={"approved": True}),
    config=config,
)
```

---

## 7. Subgraphs

Subgraphs let you **nest** one graph inside another as a node. This is essential for building complex, modular workflows.

### 7.1 When to Use Subgraphs

- Breaking a large workflow into reusable components
- Encapsulating multi-agent teams behind a single node
- Isolating state schemas (the child graph can have its own state)
- Reusing the same sub-workflow in multiple parent graphs

### 7.2 Basic Subgraph

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add


# --- Child graph ---
class ResearchState(TypedDict):
    messages: Annotated[list, add]
    research_notes: str


def search_web(state: ResearchState) -> dict:
    # ... perform web search
    return {"research_notes": "Found relevant information about..."}


def synthesize(state: ResearchState) -> dict:
    # ... synthesize findings
    return {"messages": [AIMessage(content=state["research_notes"])]}


research_builder = StateGraph(ResearchState)
research_builder.add_node("search", search_web)
research_builder.add_node("synthesize", synthesize)
research_builder.add_edge(START, "search")
research_builder.add_edge("search", "synthesize")
research_builder.add_edge("synthesize", END)
research_graph = research_builder.compile()


# --- Parent graph ---
class ParentState(TypedDict):
    messages: Annotated[list, add]
    research_notes: str


parent_builder = StateGraph(ParentState)
parent_builder.add_node("research_team", research_graph)  # subgraph as node
parent_builder.add_node("writer", write_response)

parent_builder.add_edge(START, "research_team")
parent_builder.add_edge("research_team", "writer")
parent_builder.add_edge("writer", END)

parent_graph = parent_builder.compile()
```

### 7.3 State Mapping Between Parent and Child

If the parent and child states have **different schemas**, you need to transform state at the boundary. Use a wrapper function:

```python
class ChildState(TypedDict):
    query: str
    result: str


class ParentState(TypedDict):
    messages: Annotated[list, add]
    search_result: str


def call_child_graph(state: ParentState) -> dict:
    """Adapter: transform parent state -> child state, run, transform back."""
    child_input = {
        "query": state["messages"][-1].content,
        "result": "",
    }
    child_output = child_graph.invoke(child_input)
    return {"search_result": child_output["result"]}


parent_builder.add_node("child", call_child_graph)
```

### 7.4 Subgraph Diagram

```
┌─────────────────────── Parent Graph ──────────────────────────┐
│                                                                │
│  ┌───────┐     ┌─────────────────────────────┐     ┌───────┐ │
│  │ START │────►│  research_team (subgraph)    │────►│ writer│ │
│  └───────┘     │  ┌───────┐    ┌───────────┐ │     └───┬───┘ │
│                │  │search │───►│synthesize  │ │         │     │
│                │  └───────┘    └───────────┘ │         │     │
│                └─────────────────────────────┘         │     │
│                                                    ┌───▼───┐ │
│                                                    │  END  │ │
│                                                    └───────┘ │
└────────────────────────────────────────────────────────────────┘
```

---

## 8. Multi-Agent Patterns

### 8.1 Supervisor Pattern

A **supervisor** agent decides which worker agent to invoke next. It acts as an orchestrator.

```
         ┌─────────────────────┐
         │     Supervisor       │
         │  (decides who works  │
         │   next, or finishes) │
         └──┬──────┬──────┬───┘
            │      │      │
    ┌───────▼┐  ┌──▼────┐ │
    │Research │  │Writer │ │
    │ Agent   │  │Agent  │ │
    └───┬────┘  └──┬────┘ │
        │          │      │
        └──────────┴──────┘
           (results feed back
            to supervisor)
```

```python
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from operator import add


class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add]
    next_agent: str


llm = ChatOpenAI(model="gpt-4o")


def supervisor(state: SupervisorState) -> dict:
    """The supervisor decides which agent to call next."""
    system_prompt = """You are a supervisor managing a research agent and a writer agent.
    Given the conversation, decide who should act next.
    Respond with ONLY one of: "researcher", "writer", or "FINISH"."""

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        *state["messages"],
    ])
    next_agent = response.content.strip().lower()
    return {"next_agent": next_agent, "messages": [response]}


def researcher(state: SupervisorState) -> dict:
    """Research agent gathers information."""
    response = llm.invoke([
        {"role": "system", "content": "You are a research assistant. Find relevant information."},
        *state["messages"],
    ])
    return {"messages": [HumanMessage(content=response.content, name="researcher")]}


def writer(state: SupervisorState) -> dict:
    """Writer agent produces final content."""
    response = llm.invoke([
        {"role": "system", "content": "You are a technical writer. Produce polished content."},
        *state["messages"],
    ])
    return {"messages": [HumanMessage(content=response.content, name="writer")]}


def route_supervisor(state: SupervisorState) -> str:
    next_agent = state["next_agent"]
    if next_agent == "researcher":
        return "researcher"
    elif next_agent == "writer":
        return "writer"
    else:
        return "end"


# Build the graph
builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_supervisor, {
    "researcher": "researcher",
    "writer": "writer",
    "end": END,
})
builder.add_edge("researcher", "supervisor")  # report back
builder.add_edge("writer", "supervisor")       # report back

graph = builder.compile()
```

### 8.2 Hierarchical Teams

For complex workflows, you can nest supervisors: a top-level supervisor delegates to team supervisors, each managing their own agents.

```
             ┌───────────────┐
             │ Top Supervisor │
             └──┬──────────┬─┘
                │          │
     ┌──────────▼──┐   ┌──▼──────────┐
     │ Research     │   │ Writing     │
     │ Team Super.  │   │ Team Super. │
     └──┬──────┬───┘   └──┬──────┬──┘
        │      │           │      │
     ┌──▼─┐ ┌─▼──┐     ┌──▼─┐ ┌─▼──┐
     │Web │ │Docs│     │Draft│ │Edit│
     │Srch│ │Srch│     │    │ │    │
     └────┘ └────┘     └────┘ └────┘
```

This is implemented by making each team a **subgraph** that the top supervisor uses as a node.

### 8.3 Agent Communication Through Shared State

Agents communicate by writing to shared state fields:

```python
class TeamState(TypedDict):
    messages: Annotated[list, add]
    research_findings: list[str]     # researcher writes, writer reads
    draft: str                        # writer writes, editor reads
    feedback: str                     # editor writes, writer reads
```

---

## 9. Tool Integration

### 9.1 ToolNode for Automatic Tool Execution

LangGraph provides a built-in `ToolNode` that automatically executes tool calls found in the last AI message:

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode


@tool
def search(query: str) -> str:
    """Search the web for information."""
    # In production, call a real search API
    return f"Results for: {query}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))  # simplified; use a safe evaluator in production


tools = [search, calculator]
tool_node = ToolNode(tools)

# Bind tools to the LLM so it knows they exist
llm_with_tools = ChatOpenAI(model="gpt-4o").bind_tools(tools)


def agent(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

### 9.2 Custom Tool Nodes

Sometimes you need more control than `ToolNode` provides:

```python
from langchain_core.messages import ToolMessage

def custom_tool_node(state: AgentState) -> dict:
    """Execute tool calls with custom error handling and logging."""
    last_message = state["messages"][-1]
    results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        try:
            # Look up and execute the tool
            tool_fn = {t.name: t for t in tools}[tool_name]
            result = tool_fn.invoke(tool_args)
            results.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_id,
            ))
        except Exception as e:
            # Return error as a tool message so the LLM can recover
            results.append(ToolMessage(
                content=f"Error executing {tool_name}: {str(e)}",
                tool_call_id=tool_id,
            ))

    return {"messages": results}
```

### 9.3 Error Handling in Tools

Best practice: never let a tool exception crash the graph. Return the error as a `ToolMessage` so the LLM can decide what to do:

```python
@tool
def risky_operation(param: str) -> str:
    """Perform an operation that might fail."""
    try:
        result = do_something_risky(param)
        return result
    except ValueError as e:
        return f"Error: {e}. Please try a different parameter."
```

### 9.4 Parallel Tool Execution

When the LLM returns multiple tool calls in a single response, `ToolNode` executes them all (sequentially by default). For true parallel execution:

```python
import asyncio
from langchain_core.messages import ToolMessage


async def parallel_tool_node(state: AgentState) -> dict:
    """Execute all tool calls in parallel."""
    last_message = state["messages"][-1]
    tool_map = {t.name: t for t in tools}

    async def run_tool(tc):
        try:
            result = await tool_map[tc["name"]].ainvoke(tc["args"])
            return ToolMessage(content=str(result), tool_call_id=tc["id"])
        except Exception as e:
            return ToolMessage(content=f"Error: {e}", tool_call_id=tc["id"])

    results = await asyncio.gather(
        *[run_tool(tc) for tc in last_message.tool_calls]
    )
    return {"messages": list(results)}
```

---

## 10. State Management Patterns

### 10.1 Reducers

A **reducer** controls how a state field is updated when a node returns a new value. Without a reducer, the value is **replaced**. With a reducer, the update is **merged** using the reducer function.

```python
from typing import TypedDict, Annotated
from operator import add


class AgentState(TypedDict):
    # With reducer: new messages are APPENDED to the list
    messages: Annotated[list, add]

    # Without reducer: value is REPLACED on each update
    current_step: str

    # Custom reducer: keep the maximum value
    max_score: Annotated[float, max]

    # Counter: accumulate step count
    step_count: Annotated[int, add]
```

How it works under the hood:

| Field | Node Returns | Current State | Reducer | New State |
|-------|-------------|---------------|---------|-----------|
| `messages` | `[msg3]` | `[msg1, msg2]` | `add` (list concat) | `[msg1, msg2, msg3]` |
| `current_step` | `"tools"` | `"agent"` | None (replace) | `"tools"` |
| `max_score` | `0.8` | `0.9` | `max` | `0.9` |
| `step_count` | `1` | `3` | `add` (int addition) | `4` |

### 10.2 Custom Reducers

You can define any function as a reducer:

```python
def deduplicated_add(existing: list, new: list) -> list:
    """Add items to list, avoiding duplicates."""
    seen = set(id(item) for item in existing)
    result = list(existing)
    for item in new:
        if id(item) not in seen:
            result.append(item)
            seen.add(id(item))
    return result


class AgentState(TypedDict):
    messages: Annotated[list, deduplicated_add]
```

### 10.3 Message List Management with `add_messages`

LangGraph provides a purpose-built reducer for chat messages that handles deduplication by message ID:

```python
from langgraph.graph import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

`add_messages` does the following:
- Appends new messages to the list.
- If a new message has the same `id` as an existing message, it **replaces** the existing one (useful for updating tool results).
- Handles `RemoveMessage` to delete specific messages from the history.

```python
from langchain_core.messages import RemoveMessage

def trim_messages_node(state: AgentState) -> dict:
    """Remove old messages to keep the context window manageable."""
    messages = state["messages"]
    if len(messages) > 20:
        # Remove the oldest messages, keeping the first (system) and last 10
        to_remove = messages[1:-10]
        return {
            "messages": [RemoveMessage(id=m.id) for m in to_remove]
        }
    return {}
```

### 10.4 Pydantic State Models

You can use Pydantic models for validation:

```python
from pydantic import BaseModel, Field
from langgraph.graph import add_messages


class AgentState(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    step_count: int = 0
    is_complete: bool = False

    class Config:
        arbitrary_types_allowed = True
```

---

## 11. Production Patterns

### 11.1 Streaming

LangGraph supports multiple streaming modes:

**Stream events (most granular)**

```python
async for event in graph.astream_events(
    {"messages": [HumanMessage(content="Explain quantum computing")]},
    config=config,
    version="v2",
):
    kind = event["event"]

    if kind == "on_chat_model_stream":
        # Token-by-token streaming from the LLM
        token = event["data"]["chunk"].content
        if token:
            print(token, end="", flush=True)

    elif kind == "on_chain_end":
        # A node finished execution
        print(f"\nNode completed: {event['name']}")
```

**Stream state updates**

```python
async for state_update in graph.astream(
    {"messages": [HumanMessage(content="Research LangGraph")]},
    config=config,
    stream_mode="updates",
):
    # Each update is a dict: {node_name: state_delta}
    for node_name, delta in state_update.items():
        print(f"[{node_name}] updated state with: {list(delta.keys())}")
```

**Stream full state values**

```python
async for state_snapshot in graph.astream(
    {"messages": [HumanMessage(content="...")]},
    config=config,
    stream_mode="values",
):
    # Full state at each step
    print(state_snapshot["messages"][-1].content)
```

**Multiple stream modes simultaneously**

```python
async for mode, data in graph.astream(
    inputs, config=config,
    stream_mode=["updates", "values"],
):
    if mode == "updates":
        print(f"Update: {data}")
    elif mode == "values":
        print(f"State: {data}")
```

### 11.2 Error Handling and Retry Logic

**Per-node retry with `retry` policy**

```python
from langgraph.pregel import RetryPolicy

builder.add_node(
    "api_call",
    call_external_api,
    retry=RetryPolicy(max_attempts=3, backoff_factor=2.0),
)
```

**Error routing**

```python
def call_llm_with_fallback(state: AgentState) -> dict:
    try:
        response = primary_llm.invoke(state["messages"])
        return {"messages": [response]}
    except Exception:
        response = fallback_llm.invoke(state["messages"])
        return {"messages": [response]}
```

**Global error handling with a try-except wrapper**

```python
def safe_node(func):
    """Decorator to catch exceptions in nodes and route to error handling."""
    def wrapper(state):
        try:
            return func(state)
        except Exception as e:
            return {"error": str(e), "messages": [AIMessage(content=f"Error: {e}")]}
    wrapper.__name__ = func.__name__
    return wrapper

builder.add_node("agent", safe_node(call_agent))
```

### 11.3 Logging and Tracing with LangSmith

LangSmith is the observability platform for LangChain/LangGraph:

```python
import os

# Set environment variables for LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_..."
os.environ["LANGCHAIN_PROJECT"] = "my-langgraph-project"

# All LangGraph invocations are now automatically traced
result = graph.invoke(inputs, config=config)
```

You can also add custom metadata and tags:

```python
result = graph.invoke(
    inputs,
    config={
        "configurable": {"thread_id": "t1"},
        "metadata": {"user_id": "u123", "session": "abc"},
        "tags": ["production", "v2"],
    },
)
```

### 11.4 Graph Visualization

```python
# ASCII art
graph.get_graph().print_ascii()

# Mermaid diagram (renders in Jupyter or markdown)
print(graph.get_graph().draw_mermaid())

# Save as PNG
from IPython.display import Image
img_bytes = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(img_bytes)
```

### 11.5 Testing Strategies

**Unit test individual nodes**

```python
import pytest

def test_should_continue_with_tool_calls():
    """Test the router function in isolation."""
    mock_state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "search", "args": {"q": "test"}, "id": "1"}],
            )
        ]
    }
    assert should_continue(mock_state) == "tools"


def test_should_continue_without_tool_calls():
    mock_state = {
        "messages": [AIMessage(content="Final answer")]
    }
    assert should_continue(mock_state) == "end"
```

**Integration test the full graph**

```python
def test_full_graph_execution():
    """Test the compiled graph end-to-end with a mock LLM."""
    from unittest.mock import patch, MagicMock

    mock_response = AIMessage(content="Test response")

    with patch.object(llm, "invoke", return_value=mock_response):
        result = graph.invoke({
            "messages": [HumanMessage(content="test")]
        })

    assert len(result["messages"]) == 2
    assert result["messages"][-1].content == "Test response"
```

**Test with checkpointing**

```python
def test_checkpoint_resume():
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "test-1"}}

    # First turn
    result1 = graph.invoke(
        {"messages": [HumanMessage(content="Hello")]},
        config=config,
    )

    # Second turn -- verify state continuity
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="Follow up")]},
        config=config,
    )

    assert len(result2["messages"]) == 4  # 2 human + 2 AI
```

---

## 12. Practical Examples

### 12.1 ReAct Agent with Tools

A complete ReAct agent that can search and calculate:

```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


# --- State ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# --- Tools ---
@tool
def search(query: str) -> str:
    """Search the web for current information."""
    # Replace with real search API (Tavily, SerpAPI, etc.)
    return f"Search results for '{query}': LangGraph is a framework for building stateful agent workflows."


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Example: '2 + 2'"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


tools = [search, calculator]
tool_node = ToolNode(tools)

# --- LLM ---
llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)


# --- Nodes ---
def agent(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# --- Router ---
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


# --- Build Graph ---
builder = StateGraph(AgentState)
builder.add_node("agent", agent)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "end": END,
})
builder.add_edge("tools", "agent")

# Compile with memory
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# --- Run ---
config = {"configurable": {"thread_id": "react-1"}}
result = graph.invoke(
    {"messages": [HumanMessage(content="What is LangGraph? Also, what is 42 * 17?")]},
    config=config,
)
print(result["messages"][-1].content)
```

### 12.2 Multi-Step Research Assistant

An agent that plans research, executes searches, and synthesizes results:

```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from operator import add


class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    research_plan: list[str]
    findings: Annotated[list[str], add]
    current_query_index: int
    final_report: str


llm = ChatOpenAI(model="gpt-4o", temperature=0)


def plan_research(state: ResearchState) -> dict:
    """Create a research plan from the user's question."""
    response = llm.invoke([
        SystemMessage(content=(
            "Create a research plan as a numbered list of search queries. "
            "Return ONLY the queries, one per line, numbered."
        )),
        *state["messages"],
    ])
    queries = [
        line.strip().lstrip("0123456789.)")
        for line in response.content.strip().split("\n")
        if line.strip()
    ]
    return {
        "research_plan": queries,
        "current_query_index": 0,
        "messages": [AIMessage(content=f"Research plan:\n{response.content}")],
    }


def execute_search(state: ResearchState) -> dict:
    """Execute the current search query."""
    idx = state["current_query_index"]
    query = state["research_plan"][idx]
    # In production, call a real search API
    result = f"[Search result for '{query}']: Detailed findings about {query}..."
    return {
        "findings": [result],
        "current_query_index": idx + 1,
        "messages": [AIMessage(content=f"Searched: {query}")],
    }


def synthesize_report(state: ResearchState) -> dict:
    """Synthesize all findings into a final report."""
    findings_text = "\n".join(state["findings"])
    response = llm.invoke([
        SystemMessage(content="Synthesize these research findings into a coherent report."),
        HumanMessage(content=findings_text),
    ])
    return {
        "final_report": response.content,
        "messages": [AIMessage(content=response.content)],
    }


def should_continue_research(state: ResearchState) -> str:
    if state["current_query_index"] < len(state["research_plan"]):
        return "search"
    return "synthesize"


builder = StateGraph(ResearchState)
builder.add_node("plan", plan_research)
builder.add_node("search", execute_search)
builder.add_node("synthesize", synthesize_report)

builder.add_edge(START, "plan")
builder.add_conditional_edges("plan", lambda s: "search" if s["research_plan"] else "synthesize", {
    "search": "search",
    "synthesize": "synthesize",
})
builder.add_conditional_edges("search", should_continue_research, {
    "search": "search",
    "synthesize": "synthesize",
})
builder.add_edge("synthesize", END)

research_graph = builder.compile()
```

### 12.3 Document Processing Pipeline

A pipeline that classifies documents, extracts information, and validates results:

```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages


class DocState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    document: str
    doc_type: str            # "invoice", "contract", "report", "unknown"
    extracted_data: dict
    validation_errors: list[str]
    is_valid: bool


llm = ChatOpenAI(model="gpt-4o", temperature=0)


def classify_document(state: DocState) -> dict:
    response = llm.invoke([
        SystemMessage(content=(
            "Classify this document as one of: invoice, contract, report, unknown. "
            "Return ONLY the classification word."
        )),
        HumanMessage(content=state["document"]),
    ])
    return {"doc_type": response.content.strip().lower()}


def extract_invoice(state: DocState) -> dict:
    response = llm.invoke([
        SystemMessage(content="Extract: vendor, amount, date, invoice_number from this invoice. Return as JSON."),
        HumanMessage(content=state["document"]),
    ])
    import json
    try:
        data = json.loads(response.content)
    except json.JSONDecodeError:
        data = {"raw": response.content}
    return {"extracted_data": data}


def extract_contract(state: DocState) -> dict:
    response = llm.invoke([
        SystemMessage(content="Extract: parties, effective_date, term, key_clauses. Return as JSON."),
        HumanMessage(content=state["document"]),
    ])
    import json
    try:
        data = json.loads(response.content)
    except json.JSONDecodeError:
        data = {"raw": response.content}
    return {"extracted_data": data}


def extract_generic(state: DocState) -> dict:
    response = llm.invoke([
        SystemMessage(content="Extract the key information from this document. Return as JSON."),
        HumanMessage(content=state["document"]),
    ])
    import json
    try:
        data = json.loads(response.content)
    except json.JSONDecodeError:
        data = {"raw": response.content}
    return {"extracted_data": data}


def validate(state: DocState) -> dict:
    errors = []
    data = state["extracted_data"]
    if state["doc_type"] == "invoice":
        for field in ["vendor", "amount", "date"]:
            if field not in data or not data[field]:
                errors.append(f"Missing required field: {field}")
    return {
        "validation_errors": errors,
        "is_valid": len(errors) == 0,
        "messages": [AIMessage(content=f"Validation: {'passed' if not errors else errors}")],
    }


def route_by_type(state: DocState) -> str:
    mapping = {
        "invoice": "extract_invoice",
        "contract": "extract_contract",
    }
    return mapping.get(state["doc_type"], "extract_generic")


builder = StateGraph(DocState)
builder.add_node("classify", classify_document)
builder.add_node("extract_invoice", extract_invoice)
builder.add_node("extract_contract", extract_contract)
builder.add_node("extract_generic", extract_generic)
builder.add_node("validate", validate)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_by_type, {
    "extract_invoice": "extract_invoice",
    "extract_contract": "extract_contract",
    "extract_generic": "extract_generic",
})
builder.add_edge("extract_invoice", "validate")
builder.add_edge("extract_contract", "validate")
builder.add_edge("extract_generic", "validate")
builder.add_edge("validate", END)

doc_pipeline = builder.compile()
```

### 12.4 Customer Support Bot with Escalation

```python
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt


class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    sentiment: str          # "positive", "neutral", "negative", "angry"
    escalated: bool
    resolved: bool
    ticket_id: str


llm = ChatOpenAI(model="gpt-4o", temperature=0.3)


def analyze_sentiment(state: SupportState) -> dict:
    response = llm.invoke([
        SystemMessage(content="Analyze the customer's sentiment. Return ONLY: positive, neutral, negative, or angry."),
        state["messages"][-1],
    ])
    return {"sentiment": response.content.strip().lower()}


def handle_query(state: SupportState) -> dict:
    response = llm.invoke([
        SystemMessage(content=(
            "You are a helpful customer support agent. Answer the customer's question. "
            "If you cannot resolve the issue, say 'ESCALATE' at the end."
        )),
        *state["messages"],
    ])
    return {"messages": [response]}


def check_escalation(state: SupportState) -> str:
    last_msg = state["messages"][-1].content
    sentiment = state.get("sentiment", "neutral")
    if "ESCALATE" in last_msg.upper() or sentiment == "angry":
        return "escalate"
    return "respond"


def escalate_to_human(state: SupportState) -> dict:
    """Pause the graph and wait for a human agent."""
    human_input = interrupt({
        "reason": "Customer issue requires human intervention",
        "sentiment": state["sentiment"],
        "messages": [m.content for m in state["messages"][-3:]],
    })
    return {
        "messages": [AIMessage(content=human_input["response"])],
        "escalated": True,
        "resolved": human_input.get("resolved", False),
    }


def check_resolution(state: SupportState) -> str:
    if state.get("resolved", False):
        return "end"
    return "continue"


builder = StateGraph(SupportState)
builder.add_node("sentiment", analyze_sentiment)
builder.add_node("handle", handle_query)
builder.add_node("escalate", escalate_to_human)

builder.add_edge(START, "sentiment")
builder.add_edge("sentiment", "handle")
builder.add_conditional_edges("handle", check_escalation, {
    "escalate": "escalate",
    "respond": END,
})
builder.add_conditional_edges("escalate", check_resolution, {
    "end": END,
    "continue": "sentiment",
})

memory = MemorySaver()
support_graph = builder.compile(checkpointer=memory)
```

---

## 13. Q&A Section

### Q1: What is LangGraph and how does it differ from LangChain agents?

**Answer**: LangGraph is a framework for building stateful, multi-step agent workflows as directed graphs. It is part of the LangChain ecosystem but addresses limitations of the original `AgentExecutor`:

- **AgentExecutor** runs a simple loop (LLM -> tool -> LLM -> ...) with minimal visibility. You cannot easily add branching, persistence, human-in-the-loop, or multi-agent coordination.
- **LangGraph** uses a `StateGraph` where you explicitly define nodes (functions), edges (connections), and conditional routing. This gives you full control over the execution path, built-in checkpointing, streaming, and composability through subgraphs.

LangChain's documentation itself now recommends LangGraph for production agent use cases.

---

### Q2: Explain the StateGraph model in LangGraph.

**Answer**: `StateGraph` is the core abstraction. You define:

1. **State schema** -- a `TypedDict` or Pydantic model describing all data that flows through the graph.
2. **Nodes** -- functions that receive the full state and return a partial update dict.
3. **Edges** -- connections between nodes (normal or conditional).
4. **Compile** -- produces a runnable that handles execution, state merging, checkpointing, and streaming.

After compilation, you call `graph.invoke(initial_state)` or `graph.astream(initial_state)`. The framework routes execution through nodes according to edges, merging each node's output into the running state at every step.

---

### Q3: How do conditional edges work?

**Answer**: Conditional edges let the graph branch based on the current state. You provide:

1. A **source node** name.
2. A **router function** that takes the state and returns a string key.
3. An optional **mapping dict** from string keys to target node names.

```python
graph.add_conditional_edges("agent", should_continue, {
    "tools": "tool_node",
    "end": END,
})
```

The router function inspects the state (e.g., checks if the LLM wants to call tools) and returns a key. The mapping dict resolves that key to a node name. If the mapping is omitted, the router's return value must be a valid node name or `END`.

---

### Q4: What is checkpointing and why is it important?

**Answer**: Checkpointing saves a snapshot of the full graph state after every node execution (superstep). It is important because it enables:

1. **Persistence** -- conversations survive server restarts.
2. **Time travel** -- replay from any previous checkpoint; useful for debugging and branching.
3. **Fault tolerance** -- if a node fails, you can resume from the last successful checkpoint.
4. **Human-in-the-loop** -- pause the graph, wait for human input, then resume.
5. **Multi-turn conversations** -- each `thread_id` maintains an independent conversation state.

LangGraph provides `MemorySaver` (in-memory), `SqliteSaver`, and `PostgresSaver` backends.

---

### Q5: How do you implement human-in-the-loop in LangGraph?

**Answer**: There are two mechanisms:

**Compile-time**: Pass `interrupt_before` or `interrupt_after` lists when compiling:

```python
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"],
)
```

When execution reaches the interrupted node, it pauses. The caller can inspect the state, optionally modify it with `graph.update_state()`, and resume with `graph.invoke(None, config=config)`.

**Runtime** (LangGraph 0.2.57+): Use the `interrupt()` function inside a node:

```python
from langgraph.types import interrupt, Command

def review_node(state):
    response = interrupt({"question": "Approve?"})
    if response["approved"]:
        return Command(goto="execute")
    return Command(goto="reject")
```

Resume with `graph.invoke(Command(resume={"approved": True}), config=config)`.

---

### Q6: Explain the supervisor multi-agent pattern.

**Answer**: In the supervisor pattern, one "supervisor" agent acts as an orchestrator. It receives the task, decides which specialist agent to delegate to, receives the result, and decides whether to delegate again or finish.

Implementation:
1. Define a state with shared `messages` and a `next_agent` field.
2. The supervisor node calls an LLM that outputs which agent to run next (or "FINISH").
3. Use `add_conditional_edges` to route from the supervisor to the chosen agent.
4. Each agent node writes its results to the shared messages.
5. Add edges from each agent back to the supervisor so it can decide the next step.

This pattern is great for decomposing complex tasks (e.g., research + writing + review) but adds latency due to the supervisor LLM call at each step.

---

### Q7: How does state management work with reducers?

**Answer**: By default, when a node returns `{"field": new_value}`, the field is **replaced** in the state. A **reducer** changes this behavior by defining a merge function.

```python
from typing import Annotated
from operator import add

class State(TypedDict):
    messages: Annotated[list, add]  # append instead of replace
```

When a node returns `{"messages": [new_msg]}`, the reducer (`add` = list concatenation) produces `existing_messages + [new_msg]` instead of replacing the entire list.

LangGraph also provides `add_messages` -- a purpose-built reducer that handles message deduplication by ID and supports `RemoveMessage` for trimming conversation history.

---

### Q8: What are subgraphs used for?

**Answer**: Subgraphs let you nest one compiled graph inside another as a node. Use cases:

1. **Modularity** -- break a large workflow into reusable components.
2. **Encapsulation** -- the child graph can have its own state schema, isolated from the parent.
3. **Multi-agent teams** -- each team is a subgraph with its own supervisor and workers.
4. **Reuse** -- the same subgraph can be used in multiple parent graphs.

You add a compiled subgraph as a node: `parent_builder.add_node("team", compiled_child_graph)`. If the parent and child states differ, use a wrapper function to transform state at the boundary.

---

### Q9: How do you handle errors in LangGraph?

**Answer**: Multiple strategies:

1. **Per-node retry**: `builder.add_node("api", fn, retry=RetryPolicy(max_attempts=3))`.
2. **Error as tool message**: In tool nodes, catch exceptions and return a `ToolMessage` with the error so the LLM can recover gracefully.
3. **Fallback edges**: Use conditional routing to detect errors in state and route to a fallback node.
4. **Checkpoint-based recovery**: If a node crashes, the previous checkpoint is preserved. Fix the issue and re-invoke -- execution resumes.
5. **Wrapper decorators**: Wrap node functions in try-except to catch all errors and write them to state for downstream handling.

---

### Q10: How do you test LangGraph workflows?

**Answer**:

1. **Unit test nodes**: Call node functions directly with mock state dicts and assert the returned partial state.
2. **Unit test routers**: Call router functions with mock state and assert the returned string.
3. **Integration test the graph**: Compile the graph, mock the LLM (e.g., with `unittest.mock.patch`), invoke with test inputs, and assert the final state.
4. **Test with checkpointing**: Use `MemorySaver`, invoke the graph across multiple turns, and verify state continuity.
5. **Snapshot testing**: Save graph outputs for known inputs and compare against future runs to catch regressions.

---

### Q11: What is the difference between interrupt_before and interrupt_after?

**Answer**:

- `interrupt_before=["node_name"]` -- pauses **before** the node executes. The node has not run yet. This is useful for approval workflows where you want to review what the node is about to do (e.g., inspect tool call arguments before execution).
- `interrupt_after=["node_name"]` -- pauses **after** the node executes. The node has completed and its output is in the state. This is useful for reviewing results before proceeding to the next node.

Both require a checkpointer. Resume with `graph.invoke(None, config)`.

---

### Q12: How do you stream results from a LangGraph?

**Answer**: LangGraph supports several streaming modes:

| Mode | Method | What You Get |
|------|--------|-------------|
| Events | `astream_events(version="v2")` | Granular events: token-by-token LLM output, tool calls, node starts/ends |
| Updates | `astream(stream_mode="updates")` | State delta from each node: `{node_name: partial_state}` |
| Values | `astream(stream_mode="values")` | Full state snapshot after each node |
| Multiple | `astream(stream_mode=["updates", "values"])` | Both modes simultaneously |

For production APIs, `astream_events` is the most flexible -- it lets you stream tokens to the client while also tracking which nodes execute.

---

### Q13: How does LangGraph handle concurrent or parallel execution?

**Answer**: LangGraph supports parallel execution when multiple nodes have no dependency between them. If a conditional edge routes to multiple targets simultaneously, or if you define parallel branches, LangGraph can execute those nodes in parallel.

For parallel tool calls, you can implement an async tool node that uses `asyncio.gather()` to execute all tool calls concurrently rather than sequentially.

At the graph level, the `StateGraph` execution engine runs nodes in topological order and parallelizes nodes that are at the same level of the DAG.

---

### Q14: What is the `Command` type and how is it used?

**Answer**: `Command` (from `langgraph.types`) is a special return type that gives a node explicit control over navigation and state updates:

```python
from langgraph.types import Command

def my_node(state):
    return Command(
        goto="next_node",           # explicitly specify the next node
        update={"key": "value"},    # partial state update
    )
```

`Command` can also be used to resume from an `interrupt()`:

```python
graph.invoke(Command(resume={"approved": True}), config=config)
```

It is more powerful than normal edges because the node itself decides where to go, rather than relying on external edge definitions.

---

### Q15: What is the `send()` API for fan-out parallelism?

**Answer**: `Send` lets you dynamically spawn multiple parallel executions of the same or different nodes with different inputs:

```python
from langgraph.types import Send
from langgraph.constants import START

def route_to_workers(state):
    """Fan out: send each item to a worker node with different state."""
    return [
        Send("worker", {"item": item})
        for item in state["items"]
    ]

builder.add_conditional_edges(START, route_to_workers)
```

Each `Send` creates a parallel branch. Results are merged back into the parent state using reducers. This is useful for map-reduce patterns -- e.g., processing a list of documents in parallel.

---

### Q16: How do you manage long conversation histories in LangGraph?

**Answer**: Long conversation histories can exceed LLM context windows. Strategies:

1. **Message trimming node**: Add a node that removes old messages using `RemoveMessage`:
   ```python
   from langchain_core.messages import RemoveMessage
   def trim(state):
       if len(state["messages"]) > 20:
           return {"messages": [RemoveMessage(id=m.id) for m in state["messages"][1:-10]]}
       return {}
   ```
2. **Summarization node**: Periodically summarize the conversation and replace old messages with the summary.
3. **Use `add_messages` reducer**: It correctly handles `RemoveMessage` for deletion.
4. **External memory**: Store full history in a database; pass only a window to the LLM.

---

### Q17: What is LangGraph Platform / LangGraph Cloud?

**Answer**: LangGraph Platform is a hosted/self-hosted deployment solution for LangGraph applications. It provides:

- **REST API** for graph invocation, streaming, and management
- **Persistent storage** (Postgres-backed checkpointing)
- **Task queues** for background/long-running graphs
- **Cron jobs** for scheduled graph execution
- **Monitoring dashboards** integrated with LangSmith
- **Horizontal scaling** across multiple instances

You define your graph in a `langgraph.json` manifest and deploy via `langgraph deploy` CLI or Docker.

---

### Q18: How does LangGraph compare to a simple while-loop agent?

**Answer**:

A simple while-loop agent:
```python
while True:
    response = llm.invoke(messages)
    if no_tool_calls(response):
        break
    result = execute_tools(response)
    messages.append(result)
```

This works for simple cases but lacks:
- **Persistence** -- state is lost if the process crashes.
- **Human-in-the-loop** -- no way to pause and wait for approval.
- **Visibility** -- no named steps, no graph visualization, no tracing.
- **Branching** -- hard to add conditional paths beyond "call tool or stop."
- **Multi-agent** -- adding agents means nesting more loops, which gets unmanageable.
- **Streaming** -- requires custom implementation.
- **Testing** -- each component is coupled to the loop.

LangGraph provides all of these out of the box while remaining a thin layer -- it adds structure without heavy abstraction.

---

### Q19: What is the `prebuilt` module and what does it provide?

**Answer**: `langgraph.prebuilt` provides ready-made components:

| Component | Purpose |
|-----------|---------|
| `create_react_agent()` | Builds a complete ReAct agent graph with one function call |
| `ToolNode` | Node that automatically executes tool calls from the last AI message |
| `tools_condition` | Pre-built router that checks for tool calls |
| `InjectedState` | Inject graph state into tool functions |

Example of the fastest way to create a ReAct agent:

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
tools = [search, calculator]

graph = create_react_agent(llm, tools)
result = graph.invoke({"messages": [HumanMessage(content="...")]})
```

This is equivalent to manually building the agent/tools/router graph but in one line.

---

### Q20: How do you add observability to a LangGraph application?

**Answer**: Multiple layers:

1. **LangSmith tracing**: Set `LANGCHAIN_TRACING_V2=true` and provide an API key. All invocations are automatically traced with latency, token counts, and full I/O for every node.
2. **Custom metadata**: Pass `metadata` and `tags` in the config for filtering traces.
3. **Graph visualization**: `graph.get_graph().draw_mermaid()` or `.print_ascii()` for structural understanding.
4. **Streaming events**: `astream_events` provides real-time observability of every step.
5. **Logging**: Standard Python logging in node functions.
6. **Callbacks**: LangChain callback handlers (e.g., for custom metrics) work inside LangGraph nodes.

---

### Q21: Can LangGraph nodes be async?

**Answer**: Yes. LangGraph fully supports async nodes:

```python
async def async_agent(state: AgentState) -> dict:
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

builder.add_node("agent", async_agent)
```

Use `graph.ainvoke()` or `graph.astream()` for async execution. You can mix sync and async nodes in the same graph -- LangGraph handles the interop.

---

### Q22: How do you implement rate limiting or throttling in LangGraph?

**Answer**: Since LangGraph is a framework, not a runtime, rate limiting is applied at the node level:

```python
import asyncio
from datetime import datetime, timedelta

_last_call = datetime.min

async def rate_limited_node(state: AgentState) -> dict:
    global _last_call
    min_interval = timedelta(seconds=1)  # max 1 call/sec

    elapsed = datetime.now() - _last_call
    if elapsed < min_interval:
        await asyncio.sleep((min_interval - elapsed).total_seconds())

    _last_call = datetime.now()
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}
```

For production, use a proper rate limiter (e.g., `aiolimiter`, or token bucket from your API client). You can also use LangChain's built-in rate limiting on the LLM wrapper:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", max_retries=3, request_timeout=30)
# Or configure at the API key level with OpenAI's rate limit headers
```

---

### Q23: What happens when the recursion limit is reached?

**Answer**: LangGraph raises a `GraphRecursionError`. The default `recursion_limit` is 25. This is a safety mechanism to prevent infinite loops (e.g., the agent keeps calling tools forever).

Best practices:
- Set an appropriate limit based on your use case: `config={"recursion_limit": 50}`.
- Catch the exception and return a graceful response to the user.
- Add a `step_count` field to state and a node that checks it, providing an early exit before hitting the hard limit.

---

### Q24: How do you implement memory across sessions (long-term memory)?

**Answer**: Checkpointing provides **within-session** memory (same `thread_id`). For **cross-session** memory (user preferences, facts learned over time), you need an external store:

1. **LangGraph Store** (built-in): Use `InMemoryStore` or a database-backed store to persist key-value data across threads:
   ```python
   from langgraph.store.memory import InMemoryStore
   store = InMemoryStore()
   graph = builder.compile(checkpointer=memory, store=store)
   ```
   Nodes can access the store via `config`:
   ```python
   def agent(state, config, *, store):
       user_id = config["configurable"]["user_id"]
       memories = store.search(("memories", user_id))
       # Use memories in the prompt
       ...
       # Save new memories
       store.put(("memories", user_id), "key", {"fact": "user prefers Python"})
   ```

2. **External vector store**: Use a vector database (Pinecone, Chroma, etc.) for semantic memory retrieval.

---

### Q25: Describe the full lifecycle of a single graph invocation.

**Answer**:

1. **Input**: `graph.invoke({"messages": [HumanMessage(...)]}, config={"configurable": {"thread_id": "t1"}})`.
2. **Load checkpoint**: If a checkpoint exists for `thread_id`, load it and merge the new input.
3. **Route from START**: Follow the edge from `START` to the first node.
4. **Execute node**: Call the node function with the current state. The function returns a partial update.
5. **Merge state**: Apply the partial update to the state using reducers.
6. **Save checkpoint**: Write the new state to the checkpointer.
7. **Check interrupts**: If the next node is in `interrupt_before`, pause and return.
8. **Route**: Follow edges from the current node. If conditional, call the router function.
9. **Repeat** steps 4-8 until `END` is reached or an interrupt triggers.
10. **Return**: The final state is returned to the caller.

If any node raises an exception, the state from the previous checkpoint is preserved. The caller can fix the issue and re-invoke.

---

## Quick Reference Card

```python
# Installation
# pip install langgraph langchain-openai langgraph-checkpoint-postgres

from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# 1. State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. LLM + Tools
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

# 3. Nodes
def agent(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}

# 4. Build
builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

# 5. Compile
graph = builder.compile(checkpointer=MemorySaver())

# 6. Run
result = graph.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config={"configurable": {"thread_id": "1"}},
)

# 7. Stream
async for event in graph.astream_events(inputs, config=config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```
