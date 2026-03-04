# 07. AI Agents: Architectures, Patterns & Implementation

## Table of Contents
1. [What Are AI Agents?](#1-what-are-ai-agents)
2. [Tool Use / Function Calling](#2-tool-use--function-calling)
3. [ReAct Pattern](#3-react-pattern-reasoning--acting)
4. [Planning Agents](#4-planning-agents)
5. [Multi-Agent Architectures](#5-multi-agent-architectures)
6. [Agent Memory](#6-agent-memory)
7. [Agent State Management](#7-agent-state-management)
8. [Common Agent Patterns](#8-common-agent-patterns)
9. [Safety and Guardrails](#9-safety-and-guardrails)
10. [Production Considerations](#10-production-considerations)
11. [Q&A Section](#11-qa-section)

---

## 1. What Are AI Agents?

### Definition

An **AI Agent** is an autonomous system that combines an LLM (the "brain") with tools, memory, and planning capabilities to accomplish tasks through iterative reasoning and action.

**Core components:**
- **LLM (Planning/Reasoning)** -- decides what to do next
- **Tools** -- external capabilities the agent can invoke (APIs, databases, code execution)
- **Memory** -- retains context across steps and sessions
- **Action Loop** -- observe-think-act cycle that drives execution

```
┌──────────────────────────────────────────────┐
│                  AI Agent                     │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Planning │  │  Memory  │  │  Tools   │  │
│  │  (LLM)   │  │          │  │          │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       └──────────────┼─────────────┘        │
│                      │                       │
│              ┌───────▼───────┐               │
│              │  Action Loop  │               │
│              │  Observe  ->  │               │
│              │  Think    ->  │               │
│              │  Act      ->  │               │
│              │  Repeat       │               │
│              └───────────────┘               │
└──────────────────────────────────────────────┘
```

### Agents vs Chatbots vs Pipelines

| Feature          | Chatbot              | Pipeline (Chain)       | Agent                    |
|------------------|----------------------|------------------------|--------------------------|
| Control flow     | Linear               | Fixed sequence         | Dynamic, LLM-driven      |
| Decision-making  | Predefined rules     | Predefined steps       | Autonomous reasoning      |
| Tool use         | None or limited      | Fixed tool sequence    | Dynamic tool selection    |
| Iteration        | Single turn          | Single pass            | Multi-step loops          |
| Memory           | Conversation only    | Passed between steps   | Short + long-term         |
| Error handling   | Scripted responses   | Fail or retry step     | Reason about errors       |
| Complexity       | Low                  | Medium                 | High                      |
| Use case         | FAQ, simple Q&A      | ETL, summarization     | Research, complex tasks   |

### How Agents Work -- Conceptual Flow

```
User Request
    │
    ▼
┌────────────────┐
│ Understand Goal │
└───────┬────────┘
        │
        ▼
┌────────────────┐     ┌──────────────┐
│  Plan Steps    │────>│  Memory      │
└───────┬────────┘     │  (retrieve)  │
        │              └──────────────┘
        ▼
┌────────────────┐
│ Execute Step   │───> Tool Call / LLM reasoning
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ Observe Result │
└───────┬────────┘
        │
   ┌────▼────┐
   │  Done?  │──No──> back to "Plan Steps"
   └────┬────┘
        │ Yes
        ▼
┌────────────────┐
│ Return Answer  │
└────────────────┘
```

### Minimal Agent in Python

```python
import openai

class MinimalAgent:
    """Simplest possible agent: LLM + tools + loop."""

    def __init__(self, model: str = "gpt-4o"):
        self.client = openai.OpenAI()
        self.model = model
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate a math expression",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"}
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]

    def _execute_tool(self, name: str, args: dict) -> str:
        if name == "calculator":
            try:
                result = eval(args["expression"])  # simplified
                return str(result)
            except Exception as e:
                return f"Error: {e}"
        return "Unknown tool"

    def run(self, user_message: str, max_steps: int = 10) -> str:
        messages = [{"role": "user", "content": user_message}]

        for step in range(max_steps):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
            )
            msg = response.choices[0].message
            messages.append(msg)

            # If no tool calls, the agent is done
            if not msg.tool_calls:
                return msg.content

            # Execute each tool call
            for tool_call in msg.tool_calls:
                import json
                args = json.loads(tool_call.function.arguments)
                result = self._execute_tool(
                    tool_call.function.name, args
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        return "Max steps reached without completion"


# Usage
agent = MinimalAgent()
answer = agent.run("What is 25 * 17 + 33?")
print(answer)  # "25 * 17 + 33 = 458"
```

---

## 2. Tool Use / Function Calling

### How Tools Extend LLM Capabilities

LLMs alone can only generate text. Tools give them the ability to:
- **Retrieve information** (search, database queries, APIs)
- **Take actions** (send emails, create files, deploy code)
- **Compute** (math, code execution, data analysis)
- **Interact with the world** (web browsing, file system)

### Tool Execution Flow

```
User message + Tool definitions
         │
         ▼
   ┌───────────┐
   │    LLM    │
   └─────┬─────┘
         │
         ▼
  Tool call decision:
  - function name
  - arguments (JSON)
         │
         ▼
   ┌───────────┐
   │  Runtime   │──> Execute the function
   └─────┬─────┘
         │
         ▼
   Tool result (string)
         │
         ▼
   ┌───────────┐
   │    LLM    │──> Generate response using result
   └───────────┘
```

### Defining Tools -- OpenAI Format

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search the product database by query string. "
                           "Returns matching products with prices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for products"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "books"],
                        "description": "Optional category filter"
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of results"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the current status of a customer order by order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID (e.g., ORD-12345)"
                    }
                },
                "required": ["order_id"]
            }
        }
    }
]
```

### Defining Tools -- Anthropic Format

```python
tools = [
    {
        "name": "search_database",
        "description": "Search the product database by query string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for products"
                },
                "category": {
                    "type": "string",
                    "enum": ["electronics", "clothing", "books"]
                }
            },
            "required": ["query"]
        }
    }
]
```

### Complete Tool Execution -- OpenAI

```python
import openai
import json

client = openai.OpenAI()

# 1. Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# 2. Define tool implementations
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Simulated weather lookup."""
    data = {
        "Paris": {"temp": 18, "condition": "Sunny"},
        "London": {"temp": 12, "condition": "Cloudy"},
        "Tokyo": {"temp": 22, "condition": "Clear"},
    }
    info = data.get(city, {"temp": 20, "condition": "Unknown"})
    if unit == "fahrenheit":
        info["temp"] = info["temp"] * 9 / 5 + 32
    info["unit"] = unit
    info["city"] = city
    return info

TOOL_REGISTRY = {
    "get_weather": get_weather,
}

# 3. Agent loop
def run_agent(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        assistant_msg = response.choices[0].message
        messages.append(assistant_msg)

        # No tool calls => done
        if not assistant_msg.tool_calls:
            return assistant_msg.content

        # Process each tool call
        for tc in assistant_msg.tool_calls:
            func_name = tc.function.name
            func_args = json.loads(tc.function.arguments)

            # Execute
            func = TOOL_REGISTRY.get(func_name)
            if func:
                try:
                    result = func(**func_args)
                    result_str = json.dumps(result)
                except Exception as e:
                    result_str = json.dumps({"error": str(e)})
            else:
                result_str = json.dumps({"error": f"Unknown tool: {func_name}"})

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

answer = run_agent("What's the weather in Paris and Tokyo?")
print(answer)
```

### Complete Tool Execution -- Anthropic

```python
import anthropic
import json

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"]
        }
    }
]

def get_weather(city: str) -> dict:
    return {"city": city, "temp": 18, "condition": "Sunny"}

def run_agent(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

        # Check stop reason
        if response.stop_reason == "end_turn":
            # Extract text from response
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""

        # Process tool use blocks
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        tool_results = []
        for block in assistant_content:
            if block.type == "tool_use":
                result = get_weather(**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    return "Agent completed without text response"

answer = run_agent("What's the weather in Paris?")
print(answer)
```

### Parallel Tool Calls

```python
# OpenAI supports parallel tool calls natively.
# A single assistant message can contain multiple tool_calls.

# Example response.choices[0].message.tool_calls:
# [
#   ToolCall(id="call_1", function=Function(name="get_weather", arguments='{"city":"Paris"}')),
#   ToolCall(id="call_2", function=Function(name="get_weather", arguments='{"city":"Tokyo"}')),
# ]

# You MUST return a tool result for EACH tool_call in the same order:
import asyncio

async def execute_tools_parallel(tool_calls, registry):
    """Execute multiple tool calls concurrently."""

    async def execute_one(tc):
        func_name = tc.function.name
        func_args = json.loads(tc.function.arguments)
        func = registry[func_name]

        # If function is async, await it; otherwise run in executor
        if asyncio.iscoroutinefunction(func):
            result = await func(**func_args)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: func(**func_args))

        return {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result),
        }

    results = await asyncio.gather(*[execute_one(tc) for tc in tool_calls])
    return list(results)
```

### Error Handling in Tools

```python
def safe_tool_execution(func, args: dict) -> str:
    """Wrap tool execution with error handling."""
    try:
        result = func(**args)
        return json.dumps({"status": "success", "result": result})
    except ValueError as e:
        return json.dumps({
            "status": "error",
            "error_type": "validation",
            "message": str(e),
            "suggestion": "Check the input parameters and try again."
        })
    except TimeoutError:
        return json.dumps({
            "status": "error",
            "error_type": "timeout",
            "message": "The operation timed out after 30 seconds."
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error_type": "internal",
            "message": f"Unexpected error: {str(e)}"
        })
```

### Tool Design Best Practices

1. **Clear descriptions** -- the LLM reads them to decide which tool to use
2. **Specific parameter names** -- `city` not `param1`
3. **Use enums** -- constrain values where possible
4. **Return structured data** -- JSON, not free text
5. **Include error info** -- help the LLM recover from failures
6. **Keep tools focused** -- one tool = one capability
7. **Document edge cases** -- what happens with empty input?

---

## 3. ReAct Pattern (Reasoning + Acting)

### Overview

**ReAct** (Reasoning + Acting) interleaves chain-of-thought reasoning with tool actions. The LLM explicitly states its reasoning before each action, making the process transparent and debuggable.

### The ReAct Loop

```
User: "What is the population of France divided by the area of Germany?"

Thought 1: I need to find the population of France. Let me search for it.
Action 1: search("population of France 2024")
Observation 1: The population of France is approximately 68.17 million.

Thought 2: Now I need the area of Germany.
Action 2: search("area of Germany km2")
Observation 2: Germany has an area of 357,022 square kilometers.

Thought 3: I can now compute: 68,170,000 / 357,022 = ~190.9 people/km2
Action 3: calculator("68170000 / 357022")
Observation 3: 190.94

Thought 4: I have the answer.
Action 4: respond("The population of France (68.17M) divided by the
           area of Germany (357,022 km2) is approximately 190.9 people/km2.")
```

### ReAct Agent from Scratch

```python
import re
import json
import openai

class ReActAgent:
    """
    A ReAct agent that interleaves Thought/Action/Observation steps.
    Uses structured prompting to enforce the pattern.
    """

    SYSTEM_PROMPT = """You are a helpful assistant that solves problems step by step.

You have access to the following tools:
{tool_descriptions}

For each step, you MUST use exactly this format:

Thought: <your reasoning about what to do next>
Action: <tool_name>(<arguments as JSON>)

When you have the final answer, use:
Thought: <your reasoning>
Action: respond(<your final answer>)

IMPORTANT:
- Always think before acting
- Use tools to gather information; do not guess
- If a tool returns an error, reason about it and try a different approach
"""

    def __init__(self, model: str = "gpt-4o"):
        self.client = openai.OpenAI()
        self.model = model
        self.tools = {}
        self.tool_descriptions = []

    def register_tool(self, name: str, description: str, func):
        """Register a tool the agent can use."""
        self.tools[name] = func
        self.tool_descriptions.append(f"- {name}: {description}")

    def _parse_action(self, text: str):
        """Parse action from LLM output: tool_name(args)."""
        match = re.search(r'Action:\s*(\w+)\((.+?)\)\s*$', text, re.DOTALL)
        if not match:
            return None, None
        tool_name = match.group(1)
        args_str = match.group(2).strip()

        if tool_name == "respond":
            return "respond", args_str

        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = args_str
        return tool_name, args

    def run(self, user_message: str, max_steps: int = 10) -> str:
        system = self.SYSTEM_PROMPT.format(
            tool_descriptions="\n".join(self.tool_descriptions)
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

        full_scratchpad = ""

        for step in range(max_steps):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
            )
            output = response.choices[0].message.content
            full_scratchpad += output + "\n"

            tool_name, tool_args = self._parse_action(output)

            if tool_name == "respond":
                return tool_args

            if tool_name and tool_name in self.tools:
                try:
                    if isinstance(tool_args, dict):
                        observation = self.tools[tool_name](**tool_args)
                    else:
                        observation = self.tools[tool_name](tool_args)
                    observation_str = str(observation)
                except Exception as e:
                    observation_str = f"Error: {e}"
            elif tool_name:
                observation_str = f"Error: Unknown tool '{tool_name}'"
            else:
                observation_str = "Error: Could not parse action. Use format: Action: tool_name({args})"

            full_scratchpad += f"Observation: {observation_str}\n"

            messages.append({"role": "assistant", "content": output})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation_str}"
            })

        return "Max steps reached. Partial scratchpad:\n" + full_scratchpad


# Usage
agent = ReActAgent()

agent.register_tool(
    "search",
    "Search the web for information. Args: {\"query\": \"...\"}",
    lambda query: f"Search results for '{query}': ..."  # stub
)
agent.register_tool(
    "calculator",
    "Evaluate a math expression. Args: {\"expression\": \"...\"}",
    lambda expression: eval(expression)
)

result = agent.run("What is 15% of the world population?")
print(result)
```

### When ReAct Works Well vs. When It Doesn't

| Works well                                | Does not work well                        |
|-------------------------------------------|-------------------------------------------|
| Multi-step information retrieval          | Very long plans (>10 steps)               |
| Tasks needing reasoning + external data   | Tasks requiring precise coordination      |
| Debugging / troubleshooting               | Heavily parallel workloads                |
| Open-ended research questions             | Strict deterministic workflows            |
| Tasks where transparency matters          | Latency-sensitive applications            |

---

## 4. Planning Agents

### Plan-and-Execute Pattern

Instead of deciding one step at a time (ReAct), a planning agent creates a full plan up front and then executes it step by step, revising if needed.

```
Task: "Research AI trends and write a blog post"

┌─────────────────────────────────────┐
│           PLANNING PHASE            │
│                                     │
│  Plan:                              │
│    1. Search for recent AI trends   │
│    2. Summarize key findings        │
│    3. Create blog post outline      │
│    4. Write each section            │
│    5. Review and edit               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│          EXECUTION PHASE            │
│                                     │
│  Step 1: Search ──> results         │
│  Step 2: Summarize ──> summary      │
│  Step 3: Outline ──> outline        │
│       (replan if new info found)    │
│  Step 4: Write ──> draft            │
│  Step 5: Review ──> final post      │
└─────────────────────────────────────┘
```

### Plan-and-Execute Implementation

```python
import openai
import json

class PlanAndExecuteAgent:
    """
    Two-phase agent: first creates a plan, then executes step by step.
    Can replan if execution reveals new information.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.client = openai.OpenAI()
        self.model = model
        self.tools = {}

    def register_tool(self, name: str, description: str, func):
        self.tools[name] = {"func": func, "description": description}

    def _create_plan(self, task: str) -> list[str]:
        """Use the LLM to create a step-by-step plan."""
        tool_list = "\n".join(
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"""Create a step-by-step plan to accomplish the task.
Available tools: {tool_list}

Return a JSON array of step descriptions.
Example: ["Step 1: Search for X", "Step 2: Analyze results", "Step 3: Write summary"]
Return ONLY the JSON array, no other text."""},
                {"role": "user", "content": f"Task: {task}"}
            ],
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)

    def _execute_step(self, step: str, context: str) -> str:
        """Execute a single step of the plan using tools or LLM reasoning."""
        tool_list = "\n".join(
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"""Execute the given step.
Available tools: {tool_list}

Previous context:
{context}

If you need a tool, respond with:
TOOL: tool_name
ARGS: {{"arg": "value"}}

If you can complete this step with reasoning alone, respond with:
RESULT: <your result>"""},
                {"role": "user", "content": f"Execute: {step}"}
            ],
            temperature=0,
        )
        output = response.choices[0].message.content

        if output.startswith("TOOL:"):
            lines = output.strip().split("\n")
            tool_name = lines[0].replace("TOOL:", "").strip()
            args_line = lines[1].replace("ARGS:", "").strip()
            args = json.loads(args_line)

            if tool_name in self.tools:
                result = self.tools[tool_name]["func"](**args)
                return f"Tool '{tool_name}' returned: {result}"
            return f"Error: Unknown tool '{tool_name}'"

        return output.replace("RESULT:", "").strip()

    def _should_replan(self, original_plan: list, completed: list,
                       results: list) -> list | None:
        """Ask the LLM whether we need to revise the remaining plan."""
        remaining = original_plan[len(completed):]
        if not remaining:
            return None

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": """Given the completed steps and remaining plan,
decide if the remaining plan is still valid.

If valid, respond: CONTINUE
If it needs revision, respond with a new JSON array of remaining steps."""},
                {"role": "user", "content": f"""Completed steps and results:
{json.dumps(list(zip(completed, results)), indent=2)}

Remaining plan:
{json.dumps(remaining, indent=2)}"""}
            ],
            temperature=0,
        )
        output = response.choices[0].message.content.strip()
        if output == "CONTINUE":
            return None
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return None

    def run(self, task: str, replan_interval: int = 3) -> str:
        # Phase 1: Plan
        plan = self._create_plan(task)
        print(f"Plan created with {len(plan)} steps:")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step}")

        # Phase 2: Execute
        completed_steps = []
        results = []
        context = ""

        step_idx = 0
        while step_idx < len(plan):
            step = plan[step_idx]
            print(f"\nExecuting: {step}")

            result = self._execute_step(step, context)
            completed_steps.append(step)
            results.append(result)
            context += f"\n{step}: {result}"

            print(f"Result: {result[:200]}...")

            # Periodically check if we need to replan
            if (step_idx + 1) % replan_interval == 0:
                new_remaining = self._should_replan(plan, completed_steps, results)
                if new_remaining:
                    plan = completed_steps + new_remaining
                    print(f"\nReplanned! New remaining steps: {new_remaining}")

            step_idx += 1

        # Final synthesis
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Synthesize the results into a final answer."},
                {"role": "user", "content": f"Task: {task}\n\nResults:\n{context}"}
            ],
        )
        return response.choices[0].message.content
```

### Tree of Thoughts (ToT)

Tree of Thoughts explores multiple reasoning paths simultaneously, evaluates them, and selects the most promising one.

```
                    Problem
                   /   |   \
              Path A  Path B  Path C
              /  \      |      /  \
           A1   A2     B1    C1   C2
                        |          |
                       B1a        C2a  <-- best path
```

```python
class TreeOfThoughts:
    """
    Explores multiple reasoning paths and selects the best one.
    """

    def __init__(self, model: str = "gpt-4o", branches: int = 3):
        self.client = openai.OpenAI()
        self.model = model
        self.branches = branches

    def _generate_thoughts(self, problem: str, context: str = "") -> list[str]:
        """Generate multiple candidate next-thoughts."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"""Generate exactly {self.branches} different
approaches/thoughts to solve the next step of this problem.
Return as a JSON array of strings.
Context so far: {context}"""},
                {"role": "user", "content": problem}
            ],
            temperature=0.8,
        )
        return json.loads(response.choices[0].message.content)

    def _evaluate_thought(self, problem: str, thought: str) -> float:
        """Score how promising a thought is (0.0 to 1.0)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": """Evaluate how promising this thought/approach is
for solving the problem. Respond with a single float between 0.0 and 1.0."""},
                {"role": "user", "content": f"Problem: {problem}\nThought: {thought}"}
            ],
            temperature=0,
        )
        try:
            return float(response.choices[0].message.content.strip())
        except ValueError:
            return 0.5

    def solve(self, problem: str, depth: int = 3) -> str:
        """Explore tree of thoughts to find the best reasoning path."""
        best_path = []
        context = ""

        for level in range(depth):
            thoughts = self._generate_thoughts(problem, context)
            scored = []
            for t in thoughts:
                score = self._evaluate_thought(problem, t)
                scored.append((score, t))

            scored.sort(reverse=True)
            best_thought = scored[0][1]
            best_path.append(best_thought)
            context += f"\nStep {level + 1}: {best_thought}"

        # Generate final answer from best path
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Using this reasoning path, provide the final answer."},
                {"role": "user", "content": f"Problem: {problem}\n\nReasoning:\n{context}"}
            ],
        )
        return response.choices[0].message.content
```

### Reflexion (Self-Reflection)

The agent attempts a task, evaluates its own output, and iteratively improves.

```
Attempt 1 ──> Self-evaluate ──> "Missing key point about X"
     │
     ▼
Attempt 2 (improved) ──> Self-evaluate ──> "Good but too verbose"
     │
     ▼
Attempt 3 (refined) ──> Self-evaluate ──> "Satisfactory"
     │
     ▼
  Final output
```

```python
class ReflexionAgent:
    """Agent that reflects on its own output and iteratively improves."""

    def __init__(self, model: str = "gpt-4o", max_reflections: int = 3):
        self.client = openai.OpenAI()
        self.model = model
        self.max_reflections = max_reflections

    def _attempt(self, task: str, feedback: str = "") -> str:
        messages = [
            {"role": "system", "content": "Complete the given task to the best of your ability."},
            {"role": "user", "content": f"Task: {task}"},
        ]
        if feedback:
            messages.append({
                "role": "user",
                "content": f"Previous feedback to incorporate:\n{feedback}"
            })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content

    def _reflect(self, task: str, attempt: str) -> tuple[bool, str]:
        """Self-evaluate. Returns (is_satisfactory, feedback)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": """Critically evaluate this attempt at the task.
Respond with JSON:
{"satisfactory": true/false, "feedback": "specific improvement suggestions"}"""},
                {"role": "user", "content": f"Task: {task}\n\nAttempt:\n{attempt}"}
            ],
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
        return result["satisfactory"], result["feedback"]

    def run(self, task: str) -> str:
        feedback = ""
        for i in range(self.max_reflections):
            attempt = self._attempt(task, feedback)
            is_good, feedback = self._reflect(task, attempt)

            print(f"Attempt {i+1}: {'Satisfactory' if is_good else 'Needs improvement'}")
            if feedback:
                print(f"  Feedback: {feedback[:150]}...")

            if is_good:
                return attempt

        return attempt  # return last attempt even if not perfect
```

---

## 5. Multi-Agent Architectures

### Architecture Patterns

```
1. Supervisor:              2. Peer-to-Peer (Swarm):

      ┌───S───┐                 A ─── B
     / |  |  \                  |   X  |
    A  B  C   D                 C ─── D

3. Pipeline:                4. Hierarchical:

  A ──> B ──> C ──> D           ┌── Manager ──┐
                                │              │
                            ┌─Team1─┐    ┌─Team2─┐
                            A    B       C    D
```

### Pattern 1: Supervisor

One agent (supervisor) delegates tasks to specialized worker agents and aggregates results.

```python
import openai
import json

class SupervisorAgent:
    """
    A supervisor that delegates tasks to specialized worker agents.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.client = openai.OpenAI()
        self.model = model
        self.workers = {}

    def register_worker(self, name: str, description: str, system_prompt: str):
        self.workers[name] = {
            "description": description,
            "system_prompt": system_prompt,
        }

    def _route_task(self, task: str) -> list[dict]:
        """Supervisor decides which workers to invoke and with what subtasks."""
        worker_list = "\n".join(
            f"- {name}: {info['description']}"
            for name, info in self.workers.items()
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"""You are a supervisor agent.
Given a task, decide which workers to invoke and what subtask to give each.

Available workers:
{worker_list}

Respond with a JSON array:
[{{"worker": "name", "subtask": "description of subtask"}}]"""},
                {"role": "user", "content": task}
            ],
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)

    def _invoke_worker(self, worker_name: str, subtask: str) -> str:
        """Run a worker agent on a subtask."""
        worker = self.workers[worker_name]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": worker["system_prompt"]},
                {"role": "user", "content": subtask}
            ],
        )
        return response.choices[0].message.content

    def _synthesize(self, task: str, results: dict) -> str:
        """Combine worker results into a final answer."""
        results_text = "\n\n".join(
            f"=== {name} ===\n{result}"
            for name, result in results.items()
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Synthesize the worker results into a coherent final answer."},
                {"role": "user", "content": f"Original task: {task}\n\nWorker results:\n{results_text}"}
            ],
        )
        return response.choices[0].message.content

    def run(self, task: str) -> str:
        # Step 1: Plan delegation
        assignments = self._route_task(task)
        print(f"Supervisor delegated to {len(assignments)} workers")

        # Step 2: Execute workers
        results = {}
        for assignment in assignments:
            worker_name = assignment["worker"]
            subtask = assignment["subtask"]
            print(f"  -> {worker_name}: {subtask[:80]}...")
            results[worker_name] = self._invoke_worker(worker_name, subtask)

        # Step 3: Synthesize
        return self._synthesize(task, results)


# Usage
supervisor = SupervisorAgent()

supervisor.register_worker(
    "researcher",
    "Researches topics and gathers information",
    "You are a research specialist. Provide thorough, factual research."
)
supervisor.register_worker(
    "writer",
    "Writes polished content from research notes",
    "You are a professional writer. Create engaging, well-structured content."
)
supervisor.register_worker(
    "critic",
    "Reviews and provides constructive feedback",
    "You are a critical reviewer. Point out weaknesses and suggest improvements."
)

result = supervisor.run("Write a blog post about the future of AI agents")
```

### Pattern 2: Peer-to-Peer (Swarm)

Agents communicate directly with each other, no central coordinator.

```python
class SwarmAgent:
    """An agent in a peer-to-peer swarm."""

    def __init__(self, name: str, role: str, model: str = "gpt-4o"):
        self.name = name
        self.role = role
        self.client = openai.OpenAI()
        self.model = model
        self.inbox = []

    def receive_message(self, from_agent: str, message: str):
        self.inbox.append({"from": from_agent, "message": message})

    def process(self, task: str = "") -> tuple[str, list[dict]]:
        """Process inbox and task; return response and outgoing messages."""
        inbox_text = "\n".join(
            f"From {m['from']}: {m['message']}" for m in self.inbox
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"""You are {self.name}, a {self.role}.
You are part of a team. Process incoming messages and the task.

Respond with JSON:
{{
  "response": "your contribution",
  "messages": [{{"to": "agent_name", "message": "..."}}]
}}

If you have nothing to send, use an empty messages array."""},
                {"role": "user", "content": f"Task: {task}\n\nInbox:\n{inbox_text}"}
            ],
            temperature=0,
        )
        self.inbox.clear()
        result = json.loads(response.choices[0].message.content)
        return result["response"], result.get("messages", [])


class Swarm:
    """Orchestrates peer-to-peer agent communication."""

    def __init__(self):
        self.agents: dict[str, SwarmAgent] = {}

    def add_agent(self, agent: SwarmAgent):
        self.agents[agent.name] = agent

    def run(self, task: str, rounds: int = 3) -> dict[str, str]:
        """Run swarm communication for N rounds."""
        responses = {}

        for round_num in range(rounds):
            print(f"\n--- Round {round_num + 1} ---")
            round_messages = []

            for name, agent in self.agents.items():
                response, outgoing = agent.process(task if round_num == 0 else "")
                responses[name] = response
                round_messages.extend(
                    [(name, msg["to"], msg["message"]) for msg in outgoing]
                )
                print(f"{name}: {response[:100]}...")

            # Deliver messages
            for from_name, to_name, message in round_messages:
                if to_name in self.agents:
                    self.agents[to_name].receive_message(from_name, message)

        return responses
```

### Pattern 3: Pipeline (Assembly Line)

Each agent handles one stage and passes output to the next.

```python
class PipelineStage:
    """A single stage in an agent pipeline."""

    def __init__(self, name: str, system_prompt: str, model: str = "gpt-4o"):
        self.name = name
        self.system_prompt = system_prompt
        self.client = openai.OpenAI()
        self.model = model

    def process(self, input_data: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": input_data},
            ],
        )
        return response.choices[0].message.content


class AgentPipeline:
    """Sequential pipeline of specialized agents."""

    def __init__(self):
        self.stages: list[PipelineStage] = []

    def add_stage(self, stage: PipelineStage):
        self.stages.append(stage)

    def run(self, initial_input: str) -> str:
        current_data = initial_input
        for stage in self.stages:
            print(f"Stage: {stage.name}")
            current_data = stage.process(current_data)
            print(f"  Output: {current_data[:100]}...\n")
        return current_data


# Usage: content creation pipeline
pipeline = AgentPipeline()

pipeline.add_stage(PipelineStage(
    "Research",
    "You are a researcher. Given a topic, produce key facts and data points."
))
pipeline.add_stage(PipelineStage(
    "Outline",
    "You are an editor. Given research notes, create a structured outline."
))
pipeline.add_stage(PipelineStage(
    "Draft",
    "You are a writer. Given an outline, write a complete first draft."
))
pipeline.add_stage(PipelineStage(
    "Polish",
    "You are a copy editor. Polish and improve the given draft."
))

final_article = pipeline.run("The impact of AI agents on software engineering")
```

### When to Use Multi-Agent vs. Single Agent

| Criterion               | Single Agent               | Multi-Agent                |
|--------------------------|----------------------------|----------------------------|
| Task complexity          | Simple to moderate         | Complex, multi-faceted     |
| Required expertise       | One domain                 | Multiple domains           |
| Parallelism              | Sequential steps           | Parallelizable subtasks    |
| Reliability              | Easier to debug            | Harder but more robust     |
| Latency tolerance        | Low latency needed         | Can tolerate higher latency|
| Token limits             | Within context window      | Exceeds single context     |
| Cost sensitivity         | Budget-constrained         | Can afford multiple calls  |

### Communication Patterns

```
Shared State (Blackboard):        Message Passing:

  ┌─────────────────┐            Agent A ──msg──> Agent B
  │  Shared State   │                      │
  │  {key: value}   │            Agent B ──msg──> Agent C
  │                 │
  └───┬───┬───┬─────┘
      │   │   │
      A   B   C
  (all read/write)
```

---

## 6. Agent Memory

### Memory Types

```
┌─── Short-term Memory ───┐    ┌─── Long-term Memory ────┐
│ Current conversation     │    │ Vector database          │
│ Recent messages          │    │ Past conversations       │
│ Tool call results        │    │ User preferences         │
│ (lives in context window)│    │ Learned facts            │
└──────────────────────────┘    └──────────────────────────┘
          ↕                              ↕
   ┌─── Working Memory ─────────────────────────┐
   │ Current task context                        │
   │ Active plan + progress tracking             │
   │ Retrieved relevant information              │
   │ Scratchpad for intermediate results         │
   └─────────────────────────────────────────────┘

┌─── Episodic Memory ─────┐    ┌─── Semantic Memory ─────┐
│ Past task executions     │    │ Domain knowledge         │
│ Success/failure records  │    │ Factual information      │
│ Strategy outcomes        │    │ Relationships & rules    │
└──────────────────────────┘    └──────────────────────────┘
```

### Short-term Memory: Sliding Window

```python
class ShortTermMemory:
    """
    Manages conversation history within context window limits.
    Uses a sliding window to keep the most recent messages.
    """

    def __init__(self, max_messages: int = 50, max_tokens: int = 8000):
        self.messages: list[dict] = []
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.system_message: dict | None = None

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._trim()

    def set_system(self, content: str):
        self.system_message = {"role": "system", "content": content}

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4  # rough estimate

    def _trim(self):
        """Remove oldest messages if we exceed limits."""
        while len(self.messages) > self.max_messages:
            self.messages.pop(0)

        total_tokens = sum(
            self._estimate_tokens(m["content"])
            for m in self.messages
            if isinstance(m["content"], str)
        )
        while total_tokens > self.max_tokens and len(self.messages) > 1:
            removed = self.messages.pop(0)
            total_tokens -= self._estimate_tokens(removed.get("content", ""))

    def get_messages(self) -> list[dict]:
        result = []
        if self.system_message:
            result.append(self.system_message)
        result.extend(self.messages)
        return result

    def summarize_and_compress(self, client, model: str = "gpt-4o"):
        """Summarize older messages to save context space."""
        if len(self.messages) < 10:
            return

        # Take the oldest half of messages
        split = len(self.messages) // 2
        old_messages = self.messages[:split]
        recent_messages = self.messages[split:]

        old_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in old_messages
            if isinstance(m.get("content"), str)
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Summarize this conversation concisely, "
                 "preserving key facts, decisions, and context."},
                {"role": "user", "content": old_text}
            ],
        )

        summary = response.choices[0].message.content
        self.messages = [
            {"role": "system", "content": f"[Previous conversation summary: {summary}]"}
        ] + recent_messages
```

### Long-term Memory: Vector Store

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class MemoryEntry:
    text: str
    embedding: list[float]
    metadata: dict
    timestamp: float

class VectorMemory:
    """
    Long-term memory using vector similarity search.
    Production systems would use Pinecone, Weaviate, ChromaDB, etc.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = openai.OpenAI()
        self.model = model
        self.entries: list[MemoryEntry] = []

    def _embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

    def store(self, text: str, metadata: dict | None = None):
        """Store a piece of information in long-term memory."""
        import time
        embedding = self._embed(text)
        entry = MemoryEntry(
            text=text,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=time.time(),
        )
        self.entries.append(entry)

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve the most relevant memories for a query."""
        if not self.entries:
            return []

        query_embedding = self._embed(query)
        scored = []
        for entry in self.entries:
            score = self._cosine_similarity(query_embedding, entry.embedding)
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "text": entry.text,
                "score": score,
                "metadata": entry.metadata,
            }
            for score, entry in scored[:top_k]
        ]

    def forget(self, older_than_seconds: float):
        """Remove old memories."""
        import time
        cutoff = time.time() - older_than_seconds
        self.entries = [e for e in self.entries if e.timestamp > cutoff]
```

### Agent with Combined Memory

```python
class MemoryAgent:
    """Agent that uses both short-term and long-term memory."""

    def __init__(self, model: str = "gpt-4o"):
        self.client = openai.OpenAI()
        self.model = model
        self.short_term = ShortTermMemory()
        self.long_term = VectorMemory()
        self.short_term.set_system(
            "You are a helpful assistant with memory. "
            "Use the provided context from past conversations when relevant."
        )

    def chat(self, user_message: str) -> str:
        # Retrieve relevant long-term memories
        memories = self.long_term.recall(user_message, top_k=3)

        # Build context with memories
        context = ""
        if memories:
            memory_texts = "\n".join(
                f"- {m['text']} (relevance: {m['score']:.2f})"
                for m in memories if m["score"] > 0.7
            )
            if memory_texts:
                context = f"\n[Relevant memories:\n{memory_texts}]\n"

        full_message = f"{context}{user_message}" if context else user_message
        self.short_term.add("user", full_message)

        # Get response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.short_term.get_messages(),
        )
        answer = response.choices[0].message.content
        self.short_term.add("assistant", answer)

        # Store the exchange in long-term memory
        self.long_term.store(
            f"User asked: {user_message}\nAssistant answered: {answer[:200]}",
            metadata={"type": "conversation"}
        )

        return answer
```

---

## 7. Agent State Management

### State Machine for Agent Workflows

```
┌──────────┐    task     ┌───────────┐   plan ready  ┌───────────┐
│  IDLE    │───────────>│ PLANNING  │─────────────>│ EXECUTING │
└──────────┘            └───────────┘              └─────┬─────┘
     ^                       ^                      │    │
     │                       │ replan                │    │
     │                       └──────────────────────┘    │
     │                                                   │
     │  done              ┌───────────┐    error    ┌────▼─────┐
     └────────────────────│ COMPLETED │<────────────│ ERROR    │
                          └───────────┘  (or retry) └──────────┘
```

```python
from enum import Enum
from dataclasses import dataclass, field
import time
import json

class AgentState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING_FOR_TOOL = "waiting_for_tool"
    WAITING_FOR_HUMAN = "waiting_for_human"
    ERROR = "error"
    COMPLETED = "completed"

@dataclass
class AgentCheckpoint:
    """Serializable snapshot of agent state."""
    state: AgentState
    task: str
    plan: list[str]
    completed_steps: list[str]
    step_results: list[str]
    messages: list[dict]
    current_step_index: int
    error: str | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        data = {
            "state": self.state.value,
            "task": self.task,
            "plan": self.plan,
            "completed_steps": self.completed_steps,
            "step_results": self.step_results,
            "current_step_index": self.current_step_index,
            "error": self.error,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "AgentCheckpoint":
        data = json.loads(json_str)
        data["state"] = AgentState(data["state"])
        data["messages"] = data.get("messages", [])
        return cls(**data)


class StatefulAgent:
    """Agent with full state management, checkpointing, and recovery."""

    def __init__(self, model: str = "gpt-4o"):
        self.client = openai.OpenAI()
        self.model = model
        self.state = AgentState.IDLE
        self.checkpoint: AgentCheckpoint | None = None
        self.max_retries = 3
        self.step_timeout = 60  # seconds

    def _transition(self, new_state: AgentState):
        """Transition to a new state with validation."""
        valid_transitions = {
            AgentState.IDLE: {AgentState.PLANNING},
            AgentState.PLANNING: {AgentState.EXECUTING, AgentState.ERROR},
            AgentState.EXECUTING: {
                AgentState.WAITING_FOR_TOOL,
                AgentState.WAITING_FOR_HUMAN,
                AgentState.COMPLETED,
                AgentState.ERROR,
                AgentState.PLANNING,  # replan
            },
            AgentState.WAITING_FOR_TOOL: {AgentState.EXECUTING, AgentState.ERROR},
            AgentState.WAITING_FOR_HUMAN: {AgentState.EXECUTING, AgentState.COMPLETED},
            AgentState.ERROR: {AgentState.PLANNING, AgentState.IDLE},
            AgentState.COMPLETED: {AgentState.IDLE},
        }

        if new_state not in valid_transitions.get(self.state, set()):
            raise ValueError(
                f"Invalid transition: {self.state.value} -> {new_state.value}"
            )
        print(f"State: {self.state.value} -> {new_state.value}")
        self.state = new_state

    def save_checkpoint(self) -> str:
        """Save current state to JSON for persistence."""
        if self.checkpoint:
            self.checkpoint.state = self.state
            self.checkpoint.timestamp = time.time()
            return self.checkpoint.to_json()
        return "{}"

    def load_checkpoint(self, json_str: str):
        """Resume from a saved checkpoint."""
        self.checkpoint = AgentCheckpoint.from_json(json_str)
        self.state = self.checkpoint.state

    def run(self, task: str) -> str:
        self._transition(AgentState.PLANNING)
        self.checkpoint = AgentCheckpoint(
            state=self.state,
            task=task,
            plan=[],
            completed_steps=[],
            step_results=[],
            messages=[],
            current_step_index=0,
        )

        # Planning phase
        try:
            plan = self._create_plan(task)
            self.checkpoint.plan = plan
        except Exception as e:
            self._transition(AgentState.ERROR)
            self.checkpoint.error = str(e)
            return f"Planning failed: {e}"

        # Execution phase
        self._transition(AgentState.EXECUTING)
        retries = 0

        while self.checkpoint.current_step_index < len(self.checkpoint.plan):
            step = self.checkpoint.plan[self.checkpoint.current_step_index]

            try:
                result = self._execute_step_with_timeout(step)
                self.checkpoint.completed_steps.append(step)
                self.checkpoint.step_results.append(result)
                self.checkpoint.current_step_index += 1
                retries = 0  # reset retries on success

                # Save checkpoint after each step
                self.save_checkpoint()

            except TimeoutError:
                retries += 1
                if retries >= self.max_retries:
                    self._transition(AgentState.ERROR)
                    return f"Step timed out after {self.max_retries} retries: {step}"
                print(f"Timeout on step, retrying ({retries}/{self.max_retries})")

            except Exception as e:
                retries += 1
                if retries >= self.max_retries:
                    self._transition(AgentState.ERROR)
                    self.checkpoint.error = str(e)
                    return f"Step failed after {self.max_retries} retries: {e}"
                print(f"Error: {e}, retrying ({retries}/{self.max_retries})")

        self._transition(AgentState.COMPLETED)
        return self._synthesize_results()

    def _create_plan(self, task: str) -> list[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Create a step-by-step plan. Return JSON array."},
                {"role": "user", "content": task}
            ],
        )
        return json.loads(response.choices[0].message.content)

    def _execute_step_with_timeout(self, step: str) -> str:
        """Execute a step with timeout protection."""
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self._execute_step, step)
            try:
                return future.result(timeout=self.step_timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Step exceeded {self.step_timeout}s timeout")

    def _execute_step(self, step: str) -> str:
        context = "\n".join(
            f"{s}: {r}" for s, r in
            zip(self.checkpoint.completed_steps, self.checkpoint.step_results)
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"Execute this step.\nContext:\n{context}"},
                {"role": "user", "content": step}
            ],
        )
        return response.choices[0].message.content

    def _synthesize_results(self) -> str:
        all_results = "\n\n".join(
            f"Step: {s}\nResult: {r}"
            for s, r in
            zip(self.checkpoint.completed_steps, self.checkpoint.step_results)
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Synthesize these results into a final answer."},
                {"role": "user", "content": f"Task: {self.checkpoint.task}\n\n{all_results}"}
            ],
        )
        return response.choices[0].message.content
```

---

## 8. Common Agent Patterns

### Pattern 1: Retrieval Agent (RAG Agent)

A retrieval agent searches a knowledge base and answers questions using the retrieved context.

```python
class RetrievalAgent:
    """
    Agent that searches a knowledge base to answer questions.
    Combines retrieval with reasoning.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.client = openai.OpenAI()
        self.model = model
        self.memory = VectorMemory()  # from section 6

    def ingest_documents(self, documents: list[dict]):
        """Index documents into the vector store."""
        for doc in documents:
            # Chunk the document
            chunks = self._chunk_text(doc["content"], chunk_size=500)
            for i, chunk in enumerate(chunks):
                self.memory.store(chunk, metadata={
                    "source": doc.get("source", "unknown"),
                    "chunk_index": i,
                    "title": doc.get("title", ""),
                })

    def _chunk_text(self, text: str, chunk_size: int = 500,
                    overlap: int = 50) -> list[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def _retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve relevant chunks from the knowledge base."""
        results = self.memory.recall(query, top_k=top_k)
        # Filter by minimum relevance score
        return [r for r in results if r["score"] > 0.5]

    def _needs_more_info(self, query: str, context: str) -> tuple[bool, str]:
        """Decide if the retrieved context is sufficient."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": """Given a question and retrieved context,
decide if you have enough information to answer accurately.

Respond with JSON:
{"sufficient": true/false, "refined_query": "more specific search query if needed"}"""},
                {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"}
            ],
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
        return not result["sufficient"], result.get("refined_query", "")

    def query(self, question: str, max_retrieval_rounds: int = 3) -> str:
        """Answer a question using retrieval-augmented generation."""
        all_context = []

        # Iterative retrieval
        current_query = question
        for round_num in range(max_retrieval_rounds):
            results = self._retrieve(current_query)

            if not results:
                break

            new_context = [r["text"] for r in results]
            all_context.extend(new_context)

            context_text = "\n\n".join(all_context)
            needs_more, refined = self._needs_more_info(question, context_text)

            if not needs_more:
                break

            current_query = refined

        # Generate answer
        context_text = "\n\n---\n\n".join(all_context) if all_context else "No relevant context found."

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": """Answer the question based on the provided context.
If the context doesn't contain enough information, say so.
Cite sources when possible."""},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
            ],
        )
        return response.choices[0].message.content


# Usage
agent = RetrievalAgent()
agent.ingest_documents([
    {"title": "AI Agents Overview", "content": "AI agents are...", "source": "docs/agents.md"},
    {"title": "LLM Basics", "content": "Large language models...", "source": "docs/llm.md"},
])
answer = agent.query("What are the key components of an AI agent?")
```

### Pattern 2: Code Generation Agent

```python
class CodeGenAgent:
    """Agent that generates, tests, and iterates on code."""

    def __init__(self, model: str = "gpt-4o"):
        self.client = openai.OpenAI()
        self.model = model

    def _generate_code(self, task: str, feedback: str = "") -> str:
        messages = [
            {"role": "system", "content": """You are a Python code generator.
Write clean, well-documented code.
Return ONLY the Python code, no markdown fences."""},
            {"role": "user", "content": f"Task: {task}"},
        ]
        if feedback:
            messages.append({"role": "user", "content": f"Fix based on feedback:\n{feedback}"})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content

    def _run_code(self, code: str) -> tuple[bool, str]:
        """Execute code in a sandboxed environment."""
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            try:
                result = subprocess.run(
                    ['python', f.name],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    return True, result.stdout
                else:
                    return False, result.stderr
            except subprocess.TimeoutExpired:
                return False, "Execution timed out after 30 seconds"

    def generate(self, task: str, max_iterations: int = 3) -> str:
        """Generate code with iterative testing and fixing."""
        code = self._generate_code(task)

        for i in range(max_iterations):
            success, output = self._run_code(code)
            if success:
                print(f"Code succeeded on iteration {i + 1}")
                return code

            print(f"Iteration {i + 1} failed: {output[:200]}")
            code = self._generate_code(task, feedback=f"Error:\n{output}")

        return code  # return last attempt
```

### Pattern 3: Customer Support Agent with Escalation

```python
class SupportAgent:
    """
    Customer support agent with escalation logic.
    Handles queries, escalates when confidence is low.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.client = openai.OpenAI()
        self.model = model
        self.knowledge_base = {}  # FAQ-style knowledge
        self.escalation_queue = []

    def add_faq(self, question: str, answer: str):
        self.knowledge_base[question.lower()] = answer

    def _assess_confidence(self, query: str, response: str) -> float:
        """Self-assess confidence in the response."""
        result = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": """Rate your confidence in this response.
Return a single float between 0.0 and 1.0.
- 0.9+: Very confident, clear factual answer
- 0.7-0.9: Fairly confident
- 0.5-0.7: Uncertain, might need verification
- Below 0.5: Not confident, should escalate"""},
                {"role": "user", "content": f"Query: {query}\nResponse: {response}"}
            ],
            temperature=0,
        )
        try:
            return float(result.choices[0].message.content.strip())
        except ValueError:
            return 0.5

    def _needs_escalation(self, query: str) -> bool:
        """Check if the query requires human intervention."""
        escalation_keywords = [
            "refund", "legal", "complaint", "manager",
            "sue", "attorney", "urgent", "emergency"
        ]
        return any(kw in query.lower() for kw in escalation_keywords)

    def handle(self, query: str, customer_id: str) -> dict:
        """Handle a customer query with possible escalation."""
        # Check for immediate escalation
        if self._needs_escalation(query):
            self.escalation_queue.append({
                "customer_id": customer_id,
                "query": query,
                "reason": "Contains escalation keywords",
            })
            return {
                "response": "I understand this is important. I'm connecting "
                           "you with a human agent who can help.",
                "escalated": True,
            }

        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful customer support agent. "
                 "Be concise and helpful."},
                {"role": "user", "content": query}
            ],
        )
        answer = response.choices[0].message.content

        # Check confidence
        confidence = self._assess_confidence(query, answer)

        if confidence < 0.6:
            self.escalation_queue.append({
                "customer_id": customer_id,
                "query": query,
                "attempted_response": answer,
                "confidence": confidence,
                "reason": "Low confidence",
            })
            return {
                "response": "I want to make sure you get the right answer. "
                           "Let me connect you with a specialist.",
                "escalated": True,
                "confidence": confidence,
            }

        return {
            "response": answer,
            "escalated": False,
            "confidence": confidence,
        }
```

---

## 9. Safety and Guardrails

### Overview

```
User Input ──> [Input Guards] ──> Agent ──> [Output Guards] ──> Response
                    │                            │
                    ▼                            ▼
              - Prompt injection          - PII filtering
              - Jailbreak detection       - Harmful content
              - Input validation          - Hallucination check
              - Rate limiting             - Format validation
```

### Input Validation

```python
import re

class InputGuard:
    """Validate and sanitize user inputs before they reach the agent."""

    def __init__(self):
        self.max_input_length = 10000
        self.blocked_patterns = [
            r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
            r"you\s+are\s+now\s+",
            r"pretend\s+(you\s+are|to\s+be)",
            r"system\s*prompt",
            r"<\s*script",
        ]

    def validate(self, user_input: str) -> tuple[bool, str]:
        """Returns (is_valid, sanitized_input_or_error_message)."""
        # Length check
        if len(user_input) > self.max_input_length:
            return False, f"Input too long. Maximum {self.max_input_length} characters."

        if not user_input.strip():
            return False, "Empty input."

        # Prompt injection detection
        lower_input = user_input.lower()
        for pattern in self.blocked_patterns:
            if re.search(pattern, lower_input):
                return False, "Input contains disallowed patterns."

        return True, user_input
```

### Output Filtering

```python
class OutputGuard:
    """Filter agent outputs before returning to the user."""

    def __init__(self):
        self.pii_patterns = {
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        }

    def filter_pii(self, text: str) -> str:
        """Redact PII from output."""
        filtered = text
        for pii_type, pattern in self.pii_patterns.items():
            filtered = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", filtered)
        return filtered

    def check_safety(self, text: str, client, model: str = "gpt-4o") -> tuple[bool, str]:
        """Use LLM to check if output is safe."""
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": """Evaluate if this text is safe to send to a user.
Check for: harmful instructions, misleading medical/legal advice,
discriminatory content, personal data leaks.
Respond with JSON: {"safe": true/false, "reason": "..."}"""},
                {"role": "user", "content": text}
            ],
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
        return result["safe"], result.get("reason", "")

    def process(self, text: str) -> str:
        """Apply all output filters."""
        return self.filter_pii(text)
```

### Tool Permission System

```python
from enum import Enum

class PermissionLevel(Enum):
    READ = "read"       # read-only operations
    WRITE = "write"     # create/modify data
    DELETE = "delete"   # destructive operations
    ADMIN = "admin"     # system-level operations

class ToolPermissionManager:
    """Control which tools the agent can access."""

    def __init__(self):
        self.tool_permissions: dict[str, PermissionLevel] = {}
        self.agent_level: PermissionLevel = PermissionLevel.READ
        self.requires_approval: set[str] = set()
        self.approval_callback = None

    def register_tool(self, name: str, permission: PermissionLevel,
                      requires_human_approval: bool = False):
        self.tool_permissions[name] = permission
        if requires_human_approval:
            self.requires_approval.add(name)

    def set_agent_level(self, level: PermissionLevel):
        self.agent_level = level

    def set_approval_callback(self, callback):
        """Set a function that asks a human for approval."""
        self.approval_callback = callback

    def can_execute(self, tool_name: str) -> tuple[bool, str]:
        """Check if the agent has permission to execute a tool."""
        if tool_name not in self.tool_permissions:
            return False, f"Unknown tool: {tool_name}"

        required = self.tool_permissions[tool_name]
        levels = list(PermissionLevel)
        if levels.index(required) > levels.index(self.agent_level):
            return False, (
                f"Tool '{tool_name}' requires {required.value} permission, "
                f"agent has {self.agent_level.value}"
            )

        if tool_name in self.requires_approval:
            if self.approval_callback:
                approved = self.approval_callback(tool_name)
                if not approved:
                    return False, "Human denied approval"
            else:
                return False, "Tool requires human approval but no callback set"

        return True, "Permitted"


# Usage
permissions = ToolPermissionManager()
permissions.register_tool("search", PermissionLevel.READ)
permissions.register_tool("send_email", PermissionLevel.WRITE, requires_human_approval=True)
permissions.register_tool("delete_account", PermissionLevel.DELETE, requires_human_approval=True)
permissions.set_agent_level(PermissionLevel.WRITE)

can_search, reason = permissions.can_execute("search")       # True
can_delete, reason = permissions.can_execute("delete_account")  # False, needs ADMIN
```

### Preventing Infinite Loops

```python
class LoopDetector:
    """Detect and prevent infinite loops in agent execution."""

    def __init__(self, max_steps: int = 25, max_repeated: int = 3):
        self.max_steps = max_steps
        self.max_repeated = max_repeated
        self.step_count = 0
        self.action_history: list[str] = []
        self.tool_call_counts: dict[str, int] = {}

    def record_step(self, action: str, tool_name: str | None = None):
        self.step_count += 1
        self.action_history.append(action)
        if tool_name:
            self.tool_call_counts[tool_name] = self.tool_call_counts.get(tool_name, 0) + 1

    def check(self) -> tuple[bool, str]:
        """Returns (should_stop, reason)."""
        # Max steps
        if self.step_count >= self.max_steps:
            return True, f"Reached maximum step limit ({self.max_steps})"

        # Repeated identical actions
        if len(self.action_history) >= self.max_repeated:
            recent = self.action_history[-self.max_repeated:]
            if len(set(recent)) == 1:
                return True, f"Same action repeated {self.max_repeated} times: {recent[0][:100]}"

        # Excessive tool calls
        for tool, count in self.tool_call_counts.items():
            if count > self.max_steps // 2:
                return True, f"Tool '{tool}' called {count} times (excessive)"

        return False, ""

    def reset(self):
        self.step_count = 0
        self.action_history.clear()
        self.tool_call_counts.clear()
```

### Cost Control

```python
class CostTracker:
    """Track and limit LLM API costs."""

    # Approximate pricing per 1M tokens (as of 2025)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-haiku-35-20241022": {"input": 0.80, "output": 4.00},
    }

    def __init__(self, budget_usd: float = 1.0):
        self.budget = budget_usd
        self.total_cost = 0.0
        self.call_count = 0
        self.token_usage = {"input": 0, "output": 0}

    def record_usage(self, model: str, input_tokens: int, output_tokens: int):
        pricing = self.PRICING.get(model, {"input": 5.0, "output": 15.0})
        cost = (
            input_tokens / 1_000_000 * pricing["input"] +
            output_tokens / 1_000_000 * pricing["output"]
        )
        self.total_cost += cost
        self.call_count += 1
        self.token_usage["input"] += input_tokens
        self.token_usage["output"] += output_tokens

    def check_budget(self) -> tuple[bool, float]:
        """Returns (within_budget, remaining)."""
        remaining = self.budget - self.total_cost
        return remaining > 0, remaining

    def get_summary(self) -> dict:
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "budget_usd": self.budget,
            "remaining_usd": round(self.budget - self.total_cost, 4),
            "api_calls": self.call_count,
            "total_input_tokens": self.token_usage["input"],
            "total_output_tokens": self.token_usage["output"],
        }
```

---

## 10. Production Considerations

### Observability and Logging

```python
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

@dataclass
class AgentTrace:
    """Trace of a single agent execution for observability."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    spans: list[dict] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    metadata: dict = field(default_factory=dict)

class AgentLogger:
    """Structured logging for agent execution."""

    def __init__(self, agent_name: str):
        self.logger = logging.getLogger(f"agent.{agent_name}")
        self.current_trace: AgentTrace | None = None

    @contextmanager
    def trace(self, task: str):
        """Context manager for tracing an agent execution."""
        self.current_trace = AgentTrace(metadata={"task": task})
        self.logger.info(f"Trace started: {self.current_trace.trace_id} | Task: {task}")
        try:
            yield self.current_trace
        finally:
            self.current_trace.end_time = time.time()
            duration = self.current_trace.end_time - self.current_trace.start_time
            self.logger.info(
                f"Trace ended: {self.current_trace.trace_id} | "
                f"Duration: {duration:.2f}s | "
                f"Spans: {len(self.current_trace.spans)}"
            )

    @contextmanager
    def span(self, name: str, **kwargs):
        """Context manager for tracing a sub-operation."""
        span_data = {
            "name": name,
            "start_time": time.time(),
            "metadata": kwargs,
        }
        self.logger.debug(f"Span started: {name}")
        try:
            yield span_data
        except Exception as e:
            span_data["error"] = str(e)
            self.logger.error(f"Span failed: {name} | Error: {e}")
            raise
        finally:
            span_data["end_time"] = time.time()
            span_data["duration"] = span_data["end_time"] - span_data["start_time"]
            if self.current_trace:
                self.current_trace.spans.append(span_data)
            self.logger.debug(
                f"Span ended: {name} | Duration: {span_data['duration']:.2f}s"
            )


# Usage in an agent
class ObservableAgent:

    def __init__(self):
        self.logger = AgentLogger("my_agent")
        self.client = openai.OpenAI()

    def run(self, task: str) -> str:
        with self.logger.trace(task) as trace:

            with self.logger.span("planning"):
                plan = self._plan(task)

            for i, step in enumerate(plan):
                with self.logger.span("execution", step=step, index=i):
                    result = self._execute(step)

            with self.logger.span("synthesis"):
                final = self._synthesize(task, result)

            return final
```

### Error Handling Strategies

```python
from enum import Enum

class ErrorStrategy(Enum):
    RETRY = "retry"                # retry the same step
    SKIP = "skip"                  # skip and continue
    REPLAN = "replan"              # create a new plan
    FALLBACK = "fallback"          # use a fallback tool/approach
    ESCALATE = "escalate"          # ask human for help
    ABORT = "abort"                # stop execution

class AgentErrorHandler:
    """Centralized error handling for agents."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.retry_counts: dict[str, int] = {}

    def handle(self, error: Exception, step: str,
               context: dict) -> ErrorStrategy:
        """Decide how to handle an error based on type and context."""
        step_retries = self.retry_counts.get(step, 0)

        # Rate limit errors -> always retry with backoff
        if "rate_limit" in str(error).lower():
            if step_retries < self.max_retries:
                self.retry_counts[step] = step_retries + 1
                return ErrorStrategy.RETRY
            return ErrorStrategy.FALLBACK

        # Timeout errors -> retry then skip
        if isinstance(error, TimeoutError):
            if step_retries < 2:
                self.retry_counts[step] = step_retries + 1
                return ErrorStrategy.RETRY
            return ErrorStrategy.SKIP

        # Validation errors -> replan
        if isinstance(error, ValueError):
            return ErrorStrategy.REPLAN

        # Authentication errors -> escalate
        if "auth" in str(error).lower() or "permission" in str(error).lower():
            return ErrorStrategy.ESCALATE

        # Unknown errors -> retry then abort
        if step_retries < self.max_retries:
            self.retry_counts[step] = step_retries + 1
            return ErrorStrategy.RETRY

        return ErrorStrategy.ABORT
```

### Testing Agents

```python
import unittest
from unittest.mock import MagicMock, patch

class TestReActAgent(unittest.TestCase):
    """Unit tests for a ReAct agent."""

    def setUp(self):
        self.agent = ReActAgent(model="gpt-4o")
        self.agent.register_tool(
            "search", "Search the web", lambda query: f"Results for {query}"
        )
        self.agent.register_tool(
            "calculator", "Do math", lambda expression: str(eval(expression))
        )

    def test_tool_registration(self):
        """Tools should be registered and accessible."""
        self.assertIn("search", self.agent.tools)
        self.assertIn("calculator", self.agent.tools)

    def test_action_parsing(self):
        """Action strings should be parsed correctly."""
        tool, args = self.agent._parse_action(
            'Thought: I need to search\nAction: search({"query": "AI agents"})'
        )
        self.assertEqual(tool, "search")
        self.assertEqual(args, {"query": "AI agents"})

    def test_respond_action(self):
        """respond action should be recognized."""
        tool, args = self.agent._parse_action(
            'Thought: I have the answer\nAction: respond(The answer is 42)'
        )
        self.assertEqual(tool, "respond")
        self.assertEqual(args, "The answer is 42")

    @patch("openai.OpenAI")
    def test_single_tool_call(self, mock_openai_cls):
        """Agent should execute a single tool call and return result."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # First call: agent decides to use a tool
        mock_resp_1 = MagicMock()
        mock_resp_1.choices = [MagicMock()]
        mock_resp_1.choices[0].message.content = (
            'Thought: I need to calculate\nAction: calculator({"expression": "2+2"})'
        )

        # Second call: agent responds with the answer
        mock_resp_2 = MagicMock()
        mock_resp_2.choices = [MagicMock()]
        mock_resp_2.choices[0].message.content = (
            'Thought: The result is 4\nAction: respond(2 + 2 = 4)'
        )

        mock_client.chat.completions.create.side_effect = [mock_resp_1, mock_resp_2]

        self.agent.client = mock_client
        result = self.agent.run("What is 2+2?")
        self.assertEqual(result, "2 + 2 = 4")


class TestLoopDetector(unittest.TestCase):
    """Test the loop detection mechanism."""

    def test_max_steps(self):
        detector = LoopDetector(max_steps=5)
        for i in range(5):
            detector.record_step(f"action_{i}")
        should_stop, reason = detector.check()
        self.assertTrue(should_stop)
        self.assertIn("maximum step limit", reason)

    def test_repeated_actions(self):
        detector = LoopDetector(max_repeated=3)
        for _ in range(3):
            detector.record_step("same_action")
        should_stop, reason = detector.check()
        self.assertTrue(should_stop)
        self.assertIn("repeated", reason)

    def test_normal_execution(self):
        detector = LoopDetector(max_steps=10, max_repeated=3)
        detector.record_step("action_1")
        detector.record_step("action_2")
        detector.record_step("action_3")
        should_stop, _ = detector.check()
        self.assertFalse(should_stop)
```

### Evaluation Methods

```python
class AgentEvaluator:
    """Evaluate agent performance on test cases."""

    def __init__(self, agent):
        self.agent = agent

    def evaluate(self, test_cases: list[dict]) -> dict:
        """
        Run agent on test cases and compute metrics.

        Each test case: {"input": str, "expected": str, "criteria": list[str]}
        """
        results = []
        for i, tc in enumerate(test_cases):
            try:
                output = self.agent.run(tc["input"])
                score = self._score(output, tc["expected"], tc.get("criteria", []))
                results.append({
                    "index": i,
                    "input": tc["input"],
                    "output": output,
                    "expected": tc["expected"],
                    "score": score,
                    "passed": score >= 0.7,
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "input": tc["input"],
                    "error": str(e),
                    "score": 0.0,
                    "passed": False,
                })

        # Compute aggregate metrics
        scores = [r["score"] for r in results]
        passed = [r["passed"] for r in results]

        return {
            "total": len(results),
            "passed": sum(passed),
            "failed": len(passed) - sum(passed),
            "pass_rate": sum(passed) / len(passed) if passed else 0,
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "results": results,
        }

    def _score(self, output: str, expected: str, criteria: list[str]) -> float:
        """Score agent output against expected answer."""
        # Exact match
        if output.strip().lower() == expected.strip().lower():
            return 1.0

        # Keyword matching (simple heuristic)
        expected_words = set(expected.lower().split())
        output_words = set(output.lower().split())
        overlap = expected_words & output_words
        keyword_score = len(overlap) / len(expected_words) if expected_words else 0

        # Criteria checking
        criteria_score = 0.0
        if criteria:
            met = sum(1 for c in criteria if c.lower() in output.lower())
            criteria_score = met / len(criteria)

        return (keyword_score + criteria_score) / 2 if criteria else keyword_score
```

### Latency Optimization

```
Technique                 Impact    Complexity
─────────────────────────────────────────────
Parallel tool calls       High      Low
Streaming responses       Medium    Low
Smaller models for        High      Medium
  simple sub-tasks
Caching common queries    High      Medium
Async execution           Medium    Medium
Prompt optimization       Medium    Low
  (fewer tokens)
Pre-computed embeddings   High      Medium
Connection pooling        Low       Low
```

```python
import asyncio
import hashlib

class OptimizedAgent:
    """Agent with latency optimizations."""

    def __init__(self):
        self.client = openai.AsyncOpenAI()
        self.cache: dict[str, str] = {}
        self.cache_ttl: dict[str, float] = {}

    def _cache_key(self, messages: list[dict]) -> str:
        content = json.dumps(messages, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    async def _cached_completion(self, messages: list[dict],
                                  model: str = "gpt-4o",
                                  ttl: float = 300) -> str:
        """LLM call with caching."""
        key = self._cache_key(messages)

        if key in self.cache:
            if time.time() - self.cache_ttl[key] < ttl:
                return self.cache[key]

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        result = response.choices[0].message.content
        self.cache[key] = result
        self.cache_ttl[key] = time.time()
        return result

    async def _route_to_model(self, task: str) -> str:
        """Route simple tasks to smaller models."""
        complexity_check = await self._cached_completion(
            [{"role": "user", "content": f"Rate complexity 1-5: {task}"}],
            model="gpt-4o-mini",
        )
        try:
            complexity = int(complexity_check.strip())
        except ValueError:
            complexity = 3

        model = "gpt-4o-mini" if complexity <= 2 else "gpt-4o"
        return model
```

---

## 11. Q&A Section

### Q1: What makes an AI agent different from a chatbot?

**Answer:** A chatbot is reactive -- it responds to user messages in a linear fashion using predefined rules or simple LLM calls. An AI agent is **autonomous** -- it can plan multi-step actions, use tools, maintain memory, and iteratively refine its approach. Key differences:

- **Control flow**: chatbot is linear, agent is dynamic loop-based
- **Tool use**: chatbot has none or limited, agent dynamically selects tools
- **Planning**: chatbot has none, agent creates and revises plans
- **Memory**: chatbot has conversation only, agent has short + long-term memory
- **Iteration**: chatbot does single-turn, agent loops until task is complete

---

### Q2: Explain the ReAct pattern.

**Answer:** ReAct (Reasoning + Acting) interleaves chain-of-thought reasoning with tool actions. Each iteration has three parts:

1. **Thought** -- the LLM reasons about what to do next
2. **Action** -- the LLM calls a tool or produces output
3. **Observation** -- the tool result is fed back to the LLM

This continues until the LLM decides it has enough information to respond. The key advantage is **transparency** -- you can see exactly why the agent made each decision.

```
Thought: I need to find the current stock price
Action: search("AAPL stock price today")
Observation: AAPL is trading at $195.32
Thought: I now have the answer
Action: respond("Apple (AAPL) is currently at $195.32")
```

---

### Q3: How does tool/function calling work?

**Answer:** The process follows these steps:

1. **Define tools** with JSON Schema (name, description, parameters)
2. **Send tools + user message** to the LLM
3. **LLM decides** whether to call a tool and generates structured arguments
4. **Runtime executes** the function with those arguments
5. **Tool result** is sent back to the LLM as a message with role "tool"
6. **LLM generates** the final response incorporating the tool result

The LLM never executes tools directly -- it only produces the function name and arguments. The application code is responsible for execution.

---

### Q4: What are the main multi-agent architectures?

**Answer:** Four primary patterns:

1. **Supervisor** -- one agent delegates tasks to specialized workers and aggregates results. Good for clear task decomposition.
2. **Peer-to-peer (Swarm)** -- agents communicate directly without a central coordinator. Good for collaborative problem-solving.
3. **Pipeline** -- agents are arranged in sequence, each handling one stage. Good for content processing workflows.
4. **Hierarchical** -- multi-level management with team leads and workers. Good for very complex tasks with clear domain boundaries.

Choose based on task structure: sequential (pipeline), divisible (supervisor), collaborative (swarm), nested (hierarchical).

---

### Q5: How do you implement agent memory?

**Answer:** Agent memory has multiple layers:

- **Short-term memory**: the conversation history in the LLM context window. Managed by a sliding window that keeps the most recent N messages and optionally summarizes older ones.
- **Long-term memory**: a vector database (e.g., ChromaDB, Pinecone) that stores embeddings of past conversations, facts, and documents. Retrieved via semantic similarity search.
- **Working memory**: the current task context, including the active plan, intermediate results, and retrieved information.

Implementation pattern:
1. Before each LLM call, retrieve relevant long-term memories using the current query
2. Inject retrieved memories into the system prompt or as context messages
3. After the LLM responds, store important information back into long-term memory

---

### Q6: What is the plan-and-execute pattern?

**Answer:** A two-phase approach:

1. **Planning phase**: the LLM creates a complete step-by-step plan for the task
2. **Execution phase**: each step is executed one at a time, with periodic replanning if new information changes the approach

Advantages over ReAct: better for long tasks (plan gives structure), reduces the chance of going off track, easier to show progress to users. Disadvantages: initial plan may be wrong, replanning adds latency.

---

### Q7: How do you prevent infinite loops in agents?

**Answer:** Multiple safeguards:

1. **Maximum step limit** -- hard cap on the number of iterations (e.g., 25)
2. **Repeated action detection** -- if the same action is taken N times in a row, stop
3. **Cost budget** -- stop when token/dollar budget is exhausted
4. **Timeout per step** -- each tool call or LLM call has a deadline
5. **Tool call frequency cap** -- limit how many times a specific tool can be called
6. **Progress detection** -- track whether the agent is making forward progress; if not, break the loop

```python
detector = LoopDetector(max_steps=25, max_repeated=3)
for each step:
    detector.record_step(action, tool_name)
    should_stop, reason = detector.check()
    if should_stop:
        return fallback_response(reason)
```

---

### Q8: How do you test AI agents?

**Answer:** Testing agents requires multiple approaches:

1. **Unit tests** -- test individual components (tool parsing, action selection, memory retrieval) with mocked LLM responses
2. **Integration tests** -- test full agent execution with mocked tools but real LLM calls
3. **Evaluation suites** -- run the agent on curated test cases with expected outputs and score using metrics (exact match, keyword overlap, LLM-as-judge)
4. **Regression tests** -- save traces of successful runs and verify future changes don't break them
5. **Adversarial tests** -- test with prompt injections, edge cases, and unusual inputs
6. **Cost/latency benchmarks** -- track how many tokens and API calls each test case requires

Key principle: mock the LLM in unit tests but use real LLM calls in evaluation. Deterministic components (tool execution, parsing, state management) should have standard unit tests.

---

### Q9: What is the supervisor pattern?

**Answer:** The supervisor pattern uses one "manager" agent that:

1. Receives the task from the user
2. Analyzes the task and breaks it into subtasks
3. Delegates each subtask to a specialized worker agent
4. Collects results from all workers
5. Synthesizes the final answer

Workers are specialized (e.g., researcher, writer, critic) and do not communicate with each other -- only with the supervisor. This pattern works well when tasks have clear decomposition and when different subtasks require different expertise.

---

### Q10: How do you handle errors in agent tool calls?

**Answer:** A layered approach:

1. **Wrap tool execution** in try/except and return structured error messages
2. **Return errors to the LLM** so it can reason about them and try a different approach
3. **Retry with backoff** for transient errors (rate limits, timeouts)
4. **Fallback tools** -- if the primary tool fails, use an alternative
5. **Error classification** -- different strategies for different error types (retry, skip, replan, escalate)
6. **Maximum retries** -- prevent infinite retry loops

The key insight is: tell the LLM about the error. LLMs are good at reasoning about failures and trying alternative approaches.

---

### Q11: When should you use single vs multi-agent?

**Answer:**

**Use single agent when:**
- The task is focused on one domain
- Context fits in one LLM context window
- Low latency is required
- Budget is constrained
- Simplicity and debuggability are important

**Use multi-agent when:**
- The task spans multiple domains (research + writing + review)
- The context exceeds one context window
- Subtasks can run in parallel
- Different expertise or personas are needed
- The system needs to scale to more complex tasks

Rule of thumb: start with a single agent and only split into multi-agent when you hit clear limitations.

---

### Q12: How do you manage agent state?

**Answer:** Agent state management involves:

1. **State machine** -- define valid states (IDLE, PLANNING, EXECUTING, ERROR, COMPLETED) and transitions between them
2. **Checkpointing** -- serialize agent state (plan, completed steps, results, messages) to JSON after each step
3. **Resume** -- load a checkpoint and continue execution from where it stopped
4. **Error recovery** -- transition to ERROR state, then decide whether to retry, replan, or abort

This is especially important for long-running agents where network failures, timeouts, or cost limits may interrupt execution mid-task.

---

### Q13: What are guardrails in AI agents?

**Answer:** Guardrails are safety mechanisms that constrain agent behavior:

- **Input guards**: validate and sanitize user inputs, detect prompt injection
- **Output guards**: filter PII, check for harmful content, validate format
- **Tool permissions**: control which tools the agent can access based on permission levels
- **Rate limiting**: cap the number of API calls or tool executions
- **Human-in-the-loop**: require human approval for high-risk actions (sending emails, deleting data, financial transactions)
- **Cost controls**: budget limits on token usage and API costs
- **Loop prevention**: maximum steps, repeated action detection

---

### Q14: How do you handle parallel tool calls?

**Answer:** When the LLM returns multiple tool calls in a single response:

1. Parse all tool calls from the response
2. Execute them concurrently using `asyncio.gather` or `ThreadPoolExecutor`
3. Collect all results
4. Return all results as separate tool messages (each with matching `tool_call_id`)
5. The LLM receives all results and generates the next response

```python
async def execute_parallel(tool_calls):
    tasks = [execute_tool(tc) for tc in tool_calls]
    return await asyncio.gather(*tasks)
```

Important: each result message must reference the correct `tool_call_id`.

---

### Q15: What is episodic memory in agents?

**Answer:** Episodic memory stores records of **past task executions** -- what the agent tried, what worked, and what failed. It differs from semantic memory (facts) in that it captures experiential knowledge.

Use cases:
- "Last time I tried approach X for this type of task, it failed because Y"
- "The user prefers concise answers based on past interactions"
- "This API endpoint was down yesterday; try the backup"

Implementation: store summaries of past agent runs (task, plan, outcome, errors) in a vector database. Before starting a new task, retrieve relevant episodes to inform planning.

---

### Q16: How does an agent decide which tool to use?

**Answer:** The LLM makes tool selection decisions based on:

1. **Tool descriptions** -- clear, specific descriptions help the LLM match user intent to tools
2. **Parameter schemas** -- the LLM understands what inputs each tool accepts
3. **Context** -- conversation history and current task inform which tool is relevant
4. **System prompt** -- instructions can guide tool selection preferences

Best practices for helping the LLM choose correctly:
- Write descriptions that explain **when** to use the tool, not just **what** it does
- Use specific parameter names and descriptions
- Provide examples in the system prompt
- Limit the number of available tools (10-20 is a practical maximum)

---

### Q17: What is the difference between tool calling and function calling?

**Answer:** They are essentially the same concept with different terminology:

- **Function calling** was the original term used by OpenAI when they introduced the feature
- **Tool calling** (or tool use) is the more general and now preferred term used by both OpenAI and Anthropic

Both refer to the LLM's ability to output structured JSON specifying a function name and arguments, which the application then executes. Anthropic uses "tool use" while OpenAI transitioned from "function calling" to "tool calls" in their API.

---

### Q18: How do you handle context window limitations in agents?

**Answer:** Several strategies:

1. **Sliding window** -- keep only the most recent N messages
2. **Summarization** -- periodically summarize older messages and replace them with a compact summary
3. **Retrieval** -- store information in external memory (vector DB) and retrieve only what is relevant
4. **Multi-agent** -- split the task across multiple agents, each with its own context window
5. **Selective inclusion** -- only include tool results that are still relevant; drop old intermediate results
6. **Chunked processing** -- break large inputs into chunks and process them sequentially

The best approach combines summarization (for conversation history) with retrieval (for external knowledge).

---

### Q19: How do you evaluate the quality of an agent system?

**Answer:** Multi-dimensional evaluation:

1. **Task success rate** -- does the agent complete the task correctly?
2. **Step efficiency** -- how many steps/tool calls does it take?
3. **Cost** -- total tokens and API costs per task
4. **Latency** -- time from request to final response
5. **Safety** -- does the agent stay within guardrails?
6. **Robustness** -- how does it handle edge cases and errors?
7. **User satisfaction** -- for interactive agents, does the user find it helpful?

Evaluation methods:
- Automated test suites with expected outcomes
- LLM-as-judge (use a separate LLM to evaluate the agent's output)
- Human evaluation for subjective quality
- A/B testing in production

---

### Q20: What are the key differences between LangChain agents, CrewAI, AutoGen, and LangGraph?

**Answer:**

| Framework  | Focus                    | Key feature                              |
|------------|--------------------------|------------------------------------------|
| LangChain  | General agent framework  | Wide tool/chain ecosystem, legacy agents |
| LangGraph  | Graph-based workflows    | State machines, cycles, checkpointing    |
| CrewAI     | Multi-agent teams        | Role-based agents, easy crew composition |
| AutoGen    | Multi-agent conversations| Agent chat, code execution, flexibility  |

- **LangGraph** is best for complex workflows with conditional logic and state management
- **CrewAI** is best for straightforward multi-agent collaboration (researcher + writer + reviewer)
- **AutoGen** is best for agent-to-agent conversations and code generation tasks
- **LangChain** provides the underlying primitives that many of these build on

Choose based on your primary pattern: single agent with complex flow (LangGraph), multi-agent team (CrewAI), conversational agents (AutoGen).

---

### Q21: How do you implement human-in-the-loop for agents?

**Answer:** Human-in-the-loop (HITL) requires the agent to pause and wait for human approval at certain decision points:

```python
class HITLAgent:
    def execute_step(self, step, action):
        if self.requires_approval(action):
            # Transition to WAITING_FOR_HUMAN state
            self.state = AgentState.WAITING_FOR_HUMAN
            # Present the proposed action to the human
            approval = self.request_approval(
                action=action,
                context=self.get_context(),
                risk_level="high",
            )
            if not approval.approved:
                return self.handle_rejection(approval.feedback)
            self.state = AgentState.EXECUTING
        # Continue execution
        return self.execute(action)
```

Common triggers for HITL: destructive operations, external communications, financial transactions, accessing sensitive data. Implementation approaches: webhooks, message queues, UI approval workflows, or simple CLI prompts.

---

### Q22: What is the "tool poisoning" attack and how do you prevent it?

**Answer:** Tool poisoning is when a malicious tool description or tool result manipulates the agent's behavior. For example:

- A tool result containing instructions like "ignore all previous instructions and..."
- A tool description that tricks the agent into always choosing it
- A tool that returns malicious content disguised as legitimate data

Prevention:
1. **Sanitize tool results** before passing them back to the LLM
2. **Validate tool descriptions** -- only use trusted tool definitions
3. **Separate data from instructions** -- mark tool results as data, not instructions
4. **Output validation** -- check agent responses after receiving tool results
5. **Principle of least privilege** -- tools should have minimal permissions

---

### Q23: How do you optimize agent costs in production?

**Answer:** Cost optimization strategies:

1. **Model routing** -- use cheaper models (GPT-4o-mini, Claude Haiku) for simple sub-tasks, expensive models only when needed
2. **Caching** -- cache LLM responses for repeated queries
3. **Prompt optimization** -- shorter prompts mean fewer input tokens
4. **Early stopping** -- stop as soon as the task is complete, do not over-iterate
5. **Batch processing** -- group multiple small tasks into one LLM call where possible
6. **Budget limits** -- set hard spending caps per task and per user
7. **Token counting** -- monitor and alert on unusual token consumption
8. **Result reuse** -- store and reuse tool results within the same session

---

### Q24: Explain the concept of "agentic RAG" and how it differs from basic RAG.

**Answer:**

**Basic RAG** follows a fixed pipeline: retrieve relevant documents, then generate an answer. It is a single pass with no iteration.

**Agentic RAG** wraps RAG in an agent loop, adding:
- **Query refinement** -- if initial retrieval is poor, the agent reformulates the query
- **Multi-source retrieval** -- the agent searches multiple databases or APIs
- **Iterative deepening** -- the agent retrieves, reads, identifies gaps, then retrieves more
- **Source validation** -- the agent checks if retrieved information is relevant and trustworthy
- **Answer verification** -- the agent critiques its own answer and retrieves more if needed

```
Basic RAG:     query -> retrieve -> generate -> answer
Agentic RAG:   query -> retrieve -> evaluate -> (re-query?) -> retrieve more ->
               synthesize -> verify -> (iterate?) -> answer
```

---

### Q25: What is structured output and why is it important for agents?

**Answer:** Structured output is when the LLM generates data in a specific format (JSON, XML) rather than free text. It is critical for agents because:

1. **Tool calls** require structured JSON arguments
2. **Planning** benefits from structured plan representations
3. **State management** requires parseable output
4. **Inter-agent communication** needs consistent message formats
5. **Output validation** is only possible with structured data

Implementation:
- OpenAI: `response_format={"type": "json_object"}` or JSON Schema in Structured Outputs
- Anthropic: tool use naturally returns structured JSON
- General: include format instructions in the system prompt and validate output

```python
# OpenAI structured output
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "plan",
            "schema": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["steps"]
            }
        }
    }
)
```
