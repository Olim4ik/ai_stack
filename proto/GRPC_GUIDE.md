# gRPC & Protocol Buffers Guide

A practical guide to gRPC and protobuf as used in this project.

## What Problem Does gRPC Solve?

Our services need to talk to each other. With REST/JSON, you'd write something like:

```python
# Gateway calling the embedding service via REST
response = httpx.post("http://embedding:50052/embed", json={"text": "hello"})
data = response.json()  # {"vector": [0.1, 0.2, ...], "dimensions": 1536}
```

Problems with this approach at scale:
- No contract — the server could change its response shape and the client breaks at runtime
- JSON is text-based — slow to parse, large over the wire
- No streaming — you can't send a stream of chat tokens natively
- You write client/server boilerplate by hand for every endpoint

gRPC solves all of these. You define the contract once in a `.proto` file, and everything else is auto-generated.

## The .proto File — Your Contract

A `.proto` file defines two things: **what data looks like** and **what operations exist**.

### Messages (Data Structures)

Think of these as Python dataclasses that both sides agree on:

```protobuf
message EmbedRequest {
  string text = 1;          # Field name: text, type: string
}

message EmbedResponse {
  repeated float vector = 1; # "repeated" = list of floats
  int32 dimensions = 2;
}
```

The `= 1`, `= 2` are **field numbers**, not default values. They're used for binary encoding — this is why gRPC is faster than JSON (compact binary instead of text).

#### Common types

| Protobuf type | Python equivalent | Notes |
|---------------|-------------------|-------|
| `string` | `str` | |
| `int32` | `int` | |
| `float` | `float` | |
| `bool` | `bool` | |
| `repeated X` | `list[X]` | A list/array |
| `map<K, V>` | `dict[K, V]` | A dictionary |
| `enum` | `enum.IntEnum` | Named constants |
| `Empty` (custom) | `None` | No-argument marker |

#### Example from this project — `retrieval.proto`:

```protobuf
enum SearchMode {
  DENSE = 0;      # Default value is always 0
  HYBRID = 1;
}

message SearchRequest {
  string query = 1;
  string collection = 2;
  int32 top_k = 3;
  SearchMode mode = 4;            # Enum field
  repeated Filter filters = 5;   # List of nested messages
}

message SearchResult {
  string chunk_id = 1;
  string text = 2;
  float score = 3;
  map<string, string> metadata = 4;  # Dict field
}
```

### Services (API Endpoints)

A `service` block defines RPC methods — like REST routes but typed:

```protobuf
service EmbeddingService {
  rpc Embed(EmbedRequest) returns (EmbedResponse);           # Unary: one request, one response
  rpc EmbedBatch(EmbedBatchRequest) returns (EmbedBatchResponse);  # Still unary, just bigger
  rpc GetModelInfo(Empty) returns (ModelInfo);                # No input needed
}
```

#### Streaming

The `stream` keyword enables streaming responses (like SSE but binary):

```protobuf
service AgentService {
  rpc Chat(ChatRequest) returns (stream ChatResponse);   # Server streams many responses
  rpc ResumeChat(ResumeRequest) returns (stream ChatResponse);
}
```

This is how the agent sends chat tokens one by one back to the gateway.

#### Four RPC patterns

| Pattern | Syntax | Use case |
|---------|--------|----------|
| Unary | `rpc Foo(Req) returns (Res)` | Single request, single response (most common) |
| Server streaming | `rpc Foo(Req) returns (stream Res)` | Client sends one request, server streams back multiple responses |
| Client streaming | `rpc Foo(stream Req) returns (Res)` | Client streams multiple requests, server responds once |
| Bidirectional | `rpc Foo(stream Req) returns (stream Res)` | Both sides stream |

This project uses **unary** (embedding, retrieval) and **server streaming** (agent chat).

## Code Generation — From .proto to Python

The `.proto` file is the source of truth. Python code is generated from it:

```bash
bash scripts/generate_protos.sh
```

This produces two files per proto:

```
proto/agent.proto
      |
      v
services/agent/src/generated/
      agent_pb2.py          # Message classes (ChatRequest, ChatResponse, etc.)
      agent_pb2_grpc.py     # Server base class + client stub
      __init__.py
```

### What's in each generated file

**`agent_pb2.py`** — Message classes:

```python
# Auto-generated. Use like normal Python objects:
request = agent_pb2.ChatRequest(
    session_id="abc",
    message="How do I restart auth?",
    team="platform"
)
request.message      # "How do I restart auth?"
request.session_id   # "abc"
```

**`agent_pb2_grpc.py`** — Two key classes:

```python
# 1. AgentServiceServicer — BASE CLASS for the SERVER to implement
class AgentServiceServicer:
    def Chat(self, request, context):        # You override this
        raise NotImplementedError

    def ResumeChat(self, request, context):  # You override this
        raise NotImplementedError

# 2. AgentServiceStub — CLIENT that calls the server
class AgentServiceStub:
    def __init__(self, channel): ...
    # self.Chat(request)       — call the Chat RPC
    # self.ResumeChat(request) — call the ResumeChat RPC
```

## Writing a Server (Implementing the Service)

The server **inherits** from the generated Servicer and implements the methods:

```python
# services/agent/src/service.py
import grpc
from generated import agent_pb2, agent_pb2_grpc

class AgentService(agent_pb2_grpc.AgentServiceServicer):

    async def Chat(self, request, context):
        # request is a ChatRequest object
        session_id = request.session_id
        message = request.message
        team = request.team

        # Do work... then yield streaming responses
        yield agent_pb2.ChatResponse(
            event_type="token",
            data='{"content": "To restart the auth service..."}'
        )
        yield agent_pb2.ChatResponse(
            event_type="done",
            data='{"session_id": "abc"}'
        )
```

Starting the server:

```python
# services/agent/src/main.py
import grpc

async def serve():
    server = grpc.aio.server()
    agent_pb2_grpc.add_AgentServiceServicer_to_server(AgentService(), server)
    server.add_insecure_port("[::]:50053")
    await server.start()
    await server.wait_for_termination()
```

## Writing a Client (Calling the Service)

The client uses the generated Stub with a channel:

```python
# services/gateway/src/grpc_clients/agent.py
import grpc
from generated import agent_pb2, agent_pb2_grpc

async def chat(session_id: str, message: str, team: str):
    # 1. Open a channel to the agent service
    channel = grpc.aio.insecure_channel("agent:50053")

    # 2. Create a stub (client)
    stub = agent_pb2_grpc.AgentServiceStub(channel)

    # 3. Create the request message
    request = agent_pb2.ChatRequest(
        session_id=session_id,
        message=message,
        team=team,
    )

    # 4. Call the RPC — iterate over streaming responses
    async for response in stub.Chat(request):
        print(response.event_type)  # "token", "source", "done"
        print(response.data)        # JSON string
```

## How It All Connects in This Project

```
Gateway (client)                    Agent (server)
      |                                  |
      |  channel = insecure_channel      |
      |      ("agent:50053")             |
      |                                  |
      |  stub = AgentServiceStub(ch)     |  class AgentService(AgentServiceServicer):
      |                                  |      async def Chat(self, request, context):
      |  async for resp in              |          yield ChatResponse(...)
      |      stub.Chat(ChatRequest(...)) |          yield ChatResponse(...)
      |          |                       |          yield ChatResponse(...)
      |          +--- ChatResponse 1 <---+
      |          +--- ChatResponse 2 <---+
      |          +--- ChatResponse 3 <---+
```

## Our Proto Files

### `embedding.proto` — Embedding Service (:50052)

| RPC | Input | Output | Description |
|-----|-------|--------|-------------|
| `Embed` | `EmbedRequest` (text) | `EmbedResponse` (vector, dimensions) | Embed a single text |
| `EmbedBatch` | `EmbedBatchRequest` (texts[]) | `EmbedBatchResponse` (embeddings[]) | Embed up to 64 texts at once |
| `GetModelInfo` | `Empty` | `ModelInfo` (model_name, provider, dimensions) | Check which model is loaded |

Used by: gateway (ingestion), retrieval service (query embedding)

### `retrieval.proto` — Retrieval Service (:50051)

| RPC | Input | Output | Description |
|-----|-------|--------|-------------|
| `Search` | `SearchRequest` (query, collection, top_k, mode, filters) | `SearchResponse` (results[], search_time_ms) | Semantic search over documents |
| `CreateCollection` | `CreateCollectionRequest` (name, vector_size) | `CreateCollectionResponse` (success) | Create a Qdrant collection |
| `DeleteCollection` | `DeleteCollectionRequest` (name) | `DeleteCollectionResponse` (success) | Delete a collection |
| `HealthCheck` | `Empty` | `HealthResponse` (healthy, qdrant_status) | Check Qdrant connectivity |

Used by: agent (RAG retrieval), gateway (direct search), MCP server (knowledge_search tool)

### `agent.proto` — Agent Service (:50053)

| RPC | Input | Output | Description |
|-----|-------|--------|-------------|
| `Chat` | `ChatRequest` (session_id, message, team) | `stream ChatResponse` (event_type, data) | Send a message, receive streamed events |
| `ResumeChat` | `ResumeRequest` (session_id, action_id, approved) | `stream ChatResponse` | Resume after human-in-the-loop confirmation |

Used by: gateway (proxies frontend chat requests)

## gRPC vs REST Cheat Sheet

| Concept | REST | gRPC |
|---------|------|------|
| Contract | OpenAPI/Swagger (optional) | `.proto` file (required) |
| Data format | JSON (text) | Protobuf (binary) |
| Transport | HTTP/1.1 | HTTP/2 |
| Streaming | SSE or WebSockets (bolt-on) | Native (built-in) |
| Code generation | Optional (openapi-generator) | Required (protoc) |
| Browser support | Native | Needs proxy (grpc-web) |
| Debugging | Easy (curl, Postman) | Harder (grpcurl, grpc-client) |
| Performance | Good | ~5-10x faster serialization |

**This is why the project uses both:** gRPC between services (fast, typed, streaming) and REST at the gateway (browser-compatible, easy to debug).

## Error Handling

gRPC uses status codes (similar to HTTP status codes):

```python
# Server side — returning an error
async def Chat(self, request, context):
    if not request.message:
        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        context.set_details("Message cannot be empty")
        return

# Client side — catching errors
try:
    async for response in stub.Chat(request):
        ...
except grpc.aio.AioRpcError as e:
    print(e.code())     # StatusCode.INVALID_ARGUMENT
    print(e.details())  # "Message cannot be empty"
```

Common gRPC status codes used in this project:

| Code | Meaning | HTTP equivalent |
|------|---------|-----------------|
| `OK` | Success | 200 |
| `INVALID_ARGUMENT` | Bad request | 400 |
| `NOT_FOUND` | Resource not found | 404 |
| `UNAVAILABLE` | Service down | 503 |
| `INTERNAL` | Server error | 500 |
| `UNIMPLEMENTED` | Method not coded yet | 501 |

## Quick Reference

```bash
# Generate Python code from proto files
bash scripts/generate_protos.sh

# Test a gRPC service with grpcurl (install: brew install grpcurl)
grpcurl -plaintext localhost:50052 embedding.EmbeddingService/GetModelInfo
grpcurl -plaintext -d '{"query":"auth service","collection":"team_platform","top_k":3}' \
    localhost:50051 retrieval.RetrievalService/Search

# Check if a service is running
grpcurl -plaintext localhost:50052 grpc.health.v1.Health/Check
```
