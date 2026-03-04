# 09. API Design: REST, FastAPI & gRPC

> Interview preparation for a Backend AI Engineer.
> **Primary focus: gRPC** (the user has no prior experience with it).

---

## Table of Contents

1. [REST API Fundamentals (Brief)](#1-rest-api-fundamentals-brief)
2. [gRPC Overview (DETAILED)](#2-grpc-overview)
3. [Protocol Buffers (Protobuf)](#3-protocol-buffers-protobuf)
4. [gRPC Communication Patterns](#4-grpc-communication-patterns)
5. [Python gRPC Implementation](#5-python-grpc-implementation-step-by-step)
6. [Async gRPC with Python](#6-async-grpc-with-python)
7. [gRPC Error Handling](#7-grpc-error-handling)
8. [gRPC Interceptors (Middleware)](#8-grpc-interceptors-middleware)
9. [gRPC Advanced Topics](#9-grpc-advanced-topics)
10. [gRPC + FastAPI Integration](#10-grpc--fastapi-integration)
11. [API Versioning & Authentication](#11-api-versioning--authentication)
12. [Testing gRPC](#12-testing-grpc)
13. [Q&A Section (30 Questions)](#13-qa-section)

---

## 1. REST API Fundamentals (Brief)

### 1.1 HTTP Methods

| Method   | Purpose              | Idempotent | Safe | Request Body |
|----------|----------------------|------------|------|--------------|
| `GET`    | Read a resource      | Yes        | Yes  | No           |
| `POST`   | Create a resource    | No         | No   | Yes          |
| `PUT`    | Replace a resource   | Yes        | No   | Yes          |
| `PATCH`  | Partial update       | No*        | No   | Yes          |
| `DELETE` | Remove a resource    | Yes        | No   Optional  |

> *PATCH **can** be idempotent depending on implementation, but the spec does not require it.

### 1.2 Status Codes Quick Reference

```
1xx  Informational     100 Continue
2xx  Success           200 OK, 201 Created, 204 No Content
3xx  Redirection       301 Moved Permanently, 304 Not Modified
4xx  Client Error      400 Bad Request, 401 Unauthorized, 403 Forbidden,
                       404 Not Found, 409 Conflict, 422 Unprocessable Entity,
                       429 Too Many Requests
5xx  Server Error      500 Internal Server Error, 502 Bad Gateway,
                       503 Service Unavailable, 504 Gateway Timeout
```

### 1.3 Resource Naming Conventions

```
GET    /api/v1/users              # list users
GET    /api/v1/users/42           # get user 42
POST   /api/v1/users              # create user
PUT    /api/v1/users/42           # replace user 42
PATCH  /api/v1/users/42           # partial update user 42
DELETE /api/v1/users/42           # delete user 42
GET    /api/v1/users/42/orders    # nested resource
```

**Rules**: use nouns (not verbs), plural names, lowercase, hyphens for multi-word (`/user-profiles`).

### 1.4 Versioning Strategies

| Strategy         | Example                          | Pros                | Cons                  |
|------------------|----------------------------------|---------------------|-----------------------|
| URL path         | `/api/v1/users`                  | Simple, explicit    | URL changes           |
| Query parameter  | `/api/users?version=1`           | Easy to add         | Easy to miss          |
| Header           | `Accept: application/vnd.api.v1` | Clean URLs          | Hidden, harder to test|
| Content-Type     | `Content-Type: application/vnd.api.v1+json` | Precise  | Verbose               |

### 1.5 Authentication Methods

**JWT (JSON Web Token)**:
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```
- Stateless, self-contained, includes claims (user_id, roles, exp).
- Server verifies signature without DB lookup.

**API Keys**:
```
X-API-Key: sk-abc123def456
```
- Simple, good for service-to-service communication.
- No user identity embedded; must look up in DB.

**OAuth2 Flows**:
- Authorization Code (web apps with backend)
- Client Credentials (machine-to-machine -- most common for AI backends)
- PKCE (single-page apps, mobile)

### 1.6 Rate Limiting

Common headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 57
X-RateLimit-Reset: 1672531200
Retry-After: 30
```

Algorithms: Token Bucket, Sliding Window, Fixed Window, Leaky Bucket.

### 1.7 FastAPI Quick Reference

```python
from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="User Service", version="1.0.0")

# --- Pydantic models ---
class UserCreate(BaseModel):
    name: str
    email: str
    roles: list[str] = []

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    roles: list[str]

# --- Dependency injection ---
async def get_db():
    db = Database()
    try:
        yield db
    finally:
        await db.close()

# --- Routes ---
@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db=Depends(get_db)):
    user = await db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/users", response_model=list[UserResponse])
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db=Depends(get_db),
):
    return await db.list_users(skip=skip, limit=limit)

@app.post("/users", response_model=UserResponse, status_code=201)
async def create_user(user: UserCreate, db=Depends(get_db)):
    return await db.create_user(user)
```

Key FastAPI features: automatic OpenAPI docs (`/docs`), request validation via Pydantic, dependency injection, async-first, type hints everywhere.

---

## 2. gRPC Overview

### 2.1 What is gRPC?

gRPC (**g**oogle **R**emote **P**rocedure **C**all) is a high-performance, open-source RPC framework originally developed by Google. It lets you call methods on a remote server as if they were local function calls.

**Core idea**: you define a **service** with methods and their request/response types in a `.proto` file. gRPC generates client and server code in many languages from that single file.

```
"Define once in .proto  -->  Generate code for any language  -->  Call remote methods like local functions"
```

### 2.2 Why gRPC?

| Advantage              | Explanation                                                      |
|------------------------|------------------------------------------------------------------|
| **Performance**        | Protobuf binary serialization is 5-10x faster than JSON          |
| **Type Safety**        | `.proto` is the contract; compiler catches type mismatches       |
| **Code Generation**    | Client & server stubs auto-generated for 10+ languages           |
| **HTTP/2**             | Multiplexing, header compression, persistent connections         |
| **Streaming**          | Native support for server, client, and bidirectional streaming    |
| **Deadlines**          | Built-in timeout propagation across service calls                |
| **Language Agnostic**  | Python server + Go client + Java client all from same `.proto`   |

### 2.3 gRPC vs REST -- Detailed Comparison

```
┌────────────────────┬──────────────────────┬──────────────────────────┐
│ Feature            │ REST                 │ gRPC                     │
├────────────────────┼──────────────────────┼──────────────────────────┤
│ Data Format        │ JSON (text, human-   │ Protobuf (binary,        │
│                    │ readable)            │ compact, fast)           │
├────────────────────┼──────────────────────┼──────────────────────────┤
│ Transport          │ HTTP/1.1 or HTTP/2   │ HTTP/2 only              │
├────────────────────┼──────────────────────┼──────────────────────────┤
│ Speed              │ Slower (text parse,  │ ~7-10x faster (binary    │
│                    │ no multiplexing)     │ serialization, mux)      │
├────────────────────┼──────────────────────┼──────────────────────────┤
│ API Contract       │ OpenAPI/Swagger      │ .proto file (required,   │
│                    │ (optional)           │ strongly typed)          │
├────────────────────┼──────────────────────┼──────────────────────────┤
│ Streaming          │ Limited (SSE,        │ Full bidirectional       │
│                    │ WebSocket separate)  │ streaming built-in       │
├────────────────────┼──────────────────────┼──────────────────────────┤
│ Code Generation    │ Optional (openapi-   │ Built-in (protoc)        │
│                    │ generator)           │                          │
├────────────────────┼──────────────────────┼──────────────────────────┤
│ Browser Support    │ Native               │ Via gRPC-Web proxy       │
├────────────────────┼──────────────────────┼──────────────────────────┤
│ Human Readability  │ Easy (JSON)          │ Hard (binary wire format)│
├────────────────────┼──────────────────────┼──────────────────────────┤
│ Caching            │ HTTP caching (ETag,  │ No built-in HTTP caching │
│                    │ Cache-Control)       │                          │
├────────────────────┼──────────────────────┼──────────────────────────┤
│ Error Handling     │ HTTP status codes    │ gRPC status codes +      │
│                    │                      │ rich error details       │
├────────────────────┼──────────────────────┼──────────────────────────┤
│ Learning Curve     │ Low                  │ Medium-High              │
├────────────────────┼──────────────────────┼──────────────────────────┤
│ Tooling / Debug    │ curl, Postman, any   │ grpcurl, Evans, Bloom    │
│                    │ browser              │ RPC, Postman (limited)   │
├────────────────────┼──────────────────────┼──────────────────────────┤
│ Best For           │ Public APIs, web     │ Microservices, internal  │
│                    │ frontends, CRUD      │ APIs, ML serving, IoT    │
└────────────────────┴──────────────────────┴──────────────────────────┘
```

### 2.4 When to Use Which

**Choose REST when**:
- Building a public-facing API for third-party developers
- Clients are browsers or mobile apps that need simple HTTP
- You need HTTP caching
- CRUD operations on resources
- Team is unfamiliar with gRPC

**Choose gRPC when**:
- Microservice-to-microservice communication (internal APIs)
- You need high throughput / low latency (ML model serving)
- You need streaming (real-time data, chat, live inference)
- You have a polyglot system (Python + Go + Java services)
- Strong type safety is critical
- AI/ML inference pipelines (TensorFlow Serving, Triton use gRPC)

### 2.5 gRPC Architecture

```
┌──────────────────┐         Protobuf (binary)        ┌──────────────────┐
│                  │ ◄──────────────────────────────► │                  │
│   Client (Stub)  │          HTTP/2                   │  Server (Service)│
│                  │      Multiplexed streams          │                  │
│  Auto-generated  │          Fast                     │  Auto-generated  │
│  from .proto     │                                   │  from .proto     │
└────────┬─────────┘                                   └────────┬─────────┘
         │                                                      │
         │              ┌──────────────────┐                    │
         └──────────────│   .proto file    │────────────────────┘
                        │   (contract)     │
                        │                  │
                        │  - Services      │
                        │  - Methods       │
                        │  - Messages      │
                        └──────────────────┘
                                 │
                        ┌────────┴────────┐
                        │    protoc       │
                        │  (compiler)     │
                        └────────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        Python code        Go code           Java code
        _pb2.py            .pb.go             .java
        _pb2_grpc.py       _grpc.pb.go
```

### 2.6 How a gRPC Call Works (Step by Step)

```
1. Developer writes .proto file defining services and messages
2. protoc compiler generates client stub and server skeleton
3. Developer implements server methods (business logic)
4. Client calls stub method (looks like a local function call)
5. Stub serializes request to Protobuf binary
6. Binary sent over HTTP/2 to server
7. Server deserializes request, runs method, serializes response
8. Response sent back over HTTP/2
9. Client stub deserializes response and returns it
```

### 2.7 HTTP/2 Features That Benefit gRPC

```
HTTP/1.1                          HTTP/2
─────────                         ──────
One request per connection        Multiplexing (many requests, one connection)
Text headers                      Binary + HPACK header compression
No server push                    Server push supported
No streaming                      Native bidirectional streaming
New TCP connection per request    Persistent connections
Head-of-line blocking             Stream-level flow control
```

---

## 3. Protocol Buffers (Protobuf)

Protocol Buffers are Google's language-neutral, platform-neutral mechanism for serializing structured data. Think of them as "JSON but binary, smaller, faster, and with a schema."

### 3.1 .proto File Syntax Basics

```protobuf
// Every .proto file starts with syntax declaration
syntax = "proto3";

// Package prevents naming conflicts (like Python modules)
package myapp.users;

// Optional: control Python/Go/Java package names
option go_package = "github.com/myapp/users";
option java_package = "com.myapp.users";

// Import other .proto files
import "google/protobuf/timestamp.proto";
import "google/protobuf/empty.proto";
```

### 3.2 Scalar Types

```protobuf
message ScalarExamples {
    // Integer types
    int32   small_number  = 1;   // -2^31 to 2^31-1
    int64   big_number    = 2;   // -2^63 to 2^63-1
    uint32  positive_only = 3;   // 0 to 2^32-1
    uint64  big_positive  = 4;   // 0 to 2^64-1
    sint32  signed_small  = 5;   // more efficient for negative numbers
    sint64  signed_big    = 6;

    // Floating point
    float   score         = 7;   // 32-bit IEEE 754
    double  precise_score = 8;   // 64-bit IEEE 754

    // Other
    bool    is_active     = 9;
    string  name          = 10;  // UTF-8 or 7-bit ASCII
    bytes   avatar_data   = 11;  // arbitrary byte sequence
}
```

**Python mapping**: `int32/int64` -> `int`, `float/double` -> `float`, `bool` -> `bool`, `string` -> `str`, `bytes` -> `bytes`.

### 3.3 Messages, Nested Messages, and Enums

```protobuf
syntax = "proto3";
package ecommerce;

// Enum -- first value MUST be 0
enum OrderStatus {
    ORDER_STATUS_UNSPECIFIED = 0;   // convention: ENUM_NAME_UNSPECIFIED = 0
    ORDER_STATUS_PENDING     = 1;
    ORDER_STATUS_PROCESSING  = 2;
    ORDER_STATUS_SHIPPED     = 3;
    ORDER_STATUS_DELIVERED   = 4;
    ORDER_STATUS_CANCELLED   = 5;
}

// Message with nested message
message Order {
    int64        id          = 1;
    string       customer_id = 2;
    OrderStatus  status      = 3;
    repeated Item items      = 4;   // list of items
    Address      shipping    = 5;
    double       total       = 6;

    // Nested message (only visible as Order.Item)
    message Item {
        string product_id = 1;
        string name       = 2;
        int32  quantity    = 3;
        double price      = 4;
    }

    // Nested enum
    enum Priority {
        PRIORITY_UNSPECIFIED = 0;
        PRIORITY_LOW         = 1;
        PRIORITY_HIGH        = 2;
    }
}

message Address {
    string street  = 1;
    string city    = 2;
    string state   = 3;
    string zip     = 4;
    string country = 5;
}
```

### 3.4 Repeated Fields (Lists), Maps, and Oneof

```protobuf
message UserProfile {
    string            name       = 1;
    repeated string   tags       = 2;   // list: ["ml", "python", "grpc"]
    map<string, string> metadata = 3;   // dict: {"team": "ai", "level": "senior"}

    // oneof: only one of these fields can be set at a time
    oneof contact {
        string email        = 4;
        string phone_number = 5;
        string slack_handle = 6;
    }
}
```

**Python usage**:
```python
profile = UserProfile(
    name="Alice",
    tags=["ml", "python"],         # repeated -> list
    metadata={"team": "ai"},       # map -> dict
    email="alice@example.com",     # oneof -> set one field
)

# Check which oneof field is set
print(profile.WhichOneof("contact"))  # "email"

# Repeated field operations
profile.tags.append("grpc")
profile.tags.extend(["fastapi", "docker"])

# Map operations
profile.metadata["role"] = "engineer"
```

### 3.5 Field Numbers and Backward Compatibility

**Field numbers are critical** -- they identify fields in the binary wire format.

```protobuf
message User {
    string name  = 1;   // field number 1
    string email = 2;   // field number 2
    int32  age   = 3;   // field number 3
}
```

**Rules for backward compatibility**:

| Action                        | Safe? | Explanation                                     |
|-------------------------------|-------|-------------------------------------------------|
| Add new field                 | Yes   | Old code ignores unknown fields                 |
| Remove field                  | Yes*  | Must not reuse the field number -- use `reserved`|
| Rename field                  | Yes   | Wire format uses numbers, not names              |
| Change field number           | NO    | Breaks all existing data                         |
| Change field type             | NO*   | Some compatible changes allowed (int32 -> int64) |
| Change repeated <-> singular  | NO    | Wire format differs                              |

```protobuf
message User {
    reserved 3, 8 to 10;            // reserved field numbers
    reserved "age", "old_field";    // reserved field names

    string name  = 1;
    string email = 2;
    // field 3 was "age" -- removed but number is reserved
    string phone = 4;               // new field, new number
}
```

**Field number ranges**:
- 1-15: use one byte on wire -- **use for frequently set fields**
- 16-2047: use two bytes
- 2048-536,870,911: use more bytes
- 19000-19999: reserved by protobuf implementation

### 3.6 Complete .proto Example for a User Service

```protobuf
syntax = "proto3";

package user.v1;

import "google/protobuf/timestamp.proto";
import "google/protobuf/empty.proto";
import "google/protobuf/field_mask.proto";

// ─── Messages ───

message User {
    int32  id    = 1;
    string name  = 2;
    string email = 3;
    repeated string roles = 4;
    UserStatus status     = 5;
    google.protobuf.Timestamp created_at = 6;
    google.protobuf.Timestamp updated_at = 7;
    map<string, string> metadata = 8;

    enum UserStatus {
        USER_STATUS_UNSPECIFIED = 0;
        USER_STATUS_ACTIVE      = 1;
        USER_STATUS_INACTIVE    = 2;
        USER_STATUS_SUSPENDED   = 3;
    }
}

// ─── Request / Response ───

message GetUserRequest {
    int32 id = 1;
}

message GetUserResponse {
    User user = 1;
}

message ListUsersRequest {
    int32  page_size  = 1;   // max items per page
    string page_token = 2;   // cursor for next page
    string filter     = 3;   // e.g. "status=ACTIVE"
}

message ListUsersResponse {
    repeated User users     = 1;
    string next_page_token  = 2;
}

message CreateUserRequest {
    User user = 1;
}

message UpdateUserRequest {
    User user = 1;
    google.protobuf.FieldMask update_mask = 2;  // which fields to update
}

message DeleteUserRequest {
    int32 id = 1;
}

// ─── Service ───

service UserService {
    // Unary RPCs
    rpc GetUser(GetUserRequest) returns (GetUserResponse);
    rpc CreateUser(CreateUserRequest) returns (User);
    rpc UpdateUser(UpdateUserRequest) returns (User);
    rpc DeleteUser(DeleteUserRequest) returns (google.protobuf.Empty);

    // Server streaming
    rpc ListUsers(ListUsersRequest) returns (stream User);

    // Client streaming
    rpc BulkCreateUsers(stream CreateUserRequest) returns (ListUsersResponse);

    // Bidirectional streaming
    rpc SyncUsers(stream User) returns (stream User);
}
```

---

## 4. gRPC Communication Patterns

### 4.1 Visual Overview of All Four Patterns

```
─── UNARY RPC (like a normal REST call) ───

    Client ───── request ─────► Server
    Client ◄──── response ────── Server

    Example: GetUser(id=42) -> User{...}


─── SERVER STREAMING RPC ───

    Client ───── request ─────► Server
    Client ◄════ response 1 ═══ Server
    Client ◄════ response 2 ═══ Server
    Client ◄════ response 3 ═══ Server
    Client ◄════ response N ═══ Server
    Client ◄──── END ─────────── Server

    Example: ListUsers(filter) -> stream of User objects


─── CLIENT STREAMING RPC ───

    Client ════ request 1 ════► Server
    Client ════ request 2 ════► Server
    Client ════ request N ════► Server
    Client ──── END ──────────► Server
    Client ◄──── response ────── Server

    Example: BulkUploadUsers(stream of Users) -> UploadSummary


─── BIDIRECTIONAL STREAMING RPC ───

    Client ════ request 1 ════► Server
    Client ◄════ response 1 ═══ Server
    Client ════ request 2 ════► Server
    Client ════ request 3 ════► Server
    Client ◄════ response 2 ═══ Server
    Client ◄════ response 3 ═══ Server
       ...independent streams...

    Example: Chat, real-time sync, live ML inference with feedback
```

### 4.2 When to Use Each Pattern

| Pattern              | Use Case                                         |
|----------------------|--------------------------------------------------|
| **Unary**            | Simple request/response (CRUD, auth, single query)|
| **Server streaming** | Large datasets, real-time feeds, log tailing, ML model outputting tokens |
| **Client streaming** | File upload, bulk data ingestion, sensor data     |
| **Bidirectional**    | Chat, collaborative editing, game state sync, live ML training feedback |

### 4.3 Proto Definitions for Each Pattern

```protobuf
service MLInferenceService {
    // 1. Unary: classify a single image
    rpc ClassifyImage(ImageRequest) returns (ClassificationResponse);

    // 2. Server streaming: generate text token by token (like LLM)
    rpc GenerateText(TextPrompt) returns (stream TextToken);

    // 3. Client streaming: upload chunks of a large file
    rpc UploadDataset(stream DataChunk) returns (UploadSummary);

    // 4. Bidirectional: interactive model training with live metrics
    rpc TrainModel(stream TrainingConfig) returns (stream TrainingMetrics);
}
```

---

## 5. Python gRPC Implementation (Step by Step)

### 5.1 Installation

```bash
pip install grpcio grpcio-tools

# Versions (as of writing)
# grpcio        >= 1.60.0
# grpcio-tools  >= 1.60.0
```

### 5.2 Project Structure

```
my_grpc_project/
├── protos/
│   └── user.proto           # .proto definitions
├── generated/
│   ├── __init__.py
│   ├── user_pb2.py          # generated message classes
│   └── user_pb2_grpc.py     # generated service stubs
├── server.py                # server implementation
├── client.py                # client implementation
└── requirements.txt
```

### 5.3 Compiling .proto Files

```bash
# Compile a single .proto file
python -m grpc_tools.protoc \
    -I./protos \
    --python_out=./generated \
    --grpc_python_out=./generated \
    ./protos/user.proto

# Flags explained:
#   -I./protos            : directory to search for imports
#   --python_out=         : where to put _pb2.py (message classes)
#   --grpc_python_out=    : where to put _pb2_grpc.py (service stubs)
```

### 5.4 Understanding Generated Code

**`user_pb2.py`** -- contains message classes:
```python
# Auto-generated (simplified view). DO NOT EDIT.
# Contains classes like:
#   User          -- the User message
#   GetUserRequest
#   GetUserResponse
#   ListUsersRequest
# etc.
```

**`user_pb2_grpc.py`** -- contains service stubs and servicer base classes:
```python
# Auto-generated (simplified view). DO NOT EDIT.
# Contains:
#   UserServiceStub          -- client-side stub (you call methods on this)
#   UserServiceServicer      -- server-side base class (you subclass this)
#   add_UserServiceServicer_to_server()  -- registers your servicer with server
```

### 5.5 Implementing a gRPC Server (All Four Patterns)

```python
"""server.py -- Full gRPC server with all four RPC patterns."""

import grpc
from concurrent import futures
import time
import logging

# Import generated code
import user_pb2
import user_pb2_grpc

# ─── Fake database ───
USERS_DB = {
    1: {"id": 1, "name": "Alice", "email": "alice@example.com", "roles": ["admin"]},
    2: {"id": 2, "name": "Bob", "email": "bob@example.com", "roles": ["user"]},
    3: {"id": 3, "name": "Charlie", "email": "charlie@example.com", "roles": ["user"]},
}


class UserServiceServicer(user_pb2_grpc.UserServiceServicer):
    """Implementation of the UserService defined in user.proto."""

    # ─── 1. Unary RPC ───
    def GetUser(self, request, context):
        """Single request -> single response."""
        user_data = USERS_DB.get(request.id)

        if not user_data:
            # Set gRPC error status (similar to HTTP 404)
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"User with id {request.id} not found")
            return user_pb2.GetUserResponse()

        user = user_pb2.User(
            id=user_data["id"],
            name=user_data["name"],
            email=user_data["email"],
            roles=user_data["roles"],
        )
        return user_pb2.GetUserResponse(user=user)

    # ─── 2. Server Streaming RPC ───
    def ListUsers(self, request, context):
        """Single request -> stream of responses."""
        for user_data in USERS_DB.values():
            # Each yield sends one message to the client
            user = user_pb2.User(
                id=user_data["id"],
                name=user_data["name"],
                email=user_data["email"],
                roles=user_data["roles"],
            )
            yield user
            time.sleep(0.1)  # simulate delay between items

    # ─── 3. Client Streaming RPC ───
    def BulkCreateUsers(self, request_iterator, context):
        """Stream of requests -> single response."""
        created = []
        for req in request_iterator:
            # Each iteration receives one message from the client
            new_id = max(USERS_DB.keys()) + 1 if USERS_DB else 1
            USERS_DB[new_id] = {
                "id": new_id,
                "name": req.user.name,
                "email": req.user.email,
                "roles": list(req.user.roles),
            }
            created.append(user_pb2.User(id=new_id, name=req.user.name))

        return user_pb2.ListUsersResponse(users=created)

    # ─── 4. Bidirectional Streaming RPC ───
    def SyncUsers(self, request_iterator, context):
        """Stream of requests <-> stream of responses (independent)."""
        for user in request_iterator:
            # Process each incoming user and immediately respond
            processed = user_pb2.User(
                id=user.id,
                name=user.name.upper(),  # transform: uppercase
                email=user.email,
            )
            yield processed


def serve():
    """Start the gRPC server."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),    # 50MB
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50MB
        ],
    )

    # Register servicer
    user_pb2_grpc.add_UserServiceServicer_to_server(
        UserServiceServicer(), server
    )

    # Listen on port 50051
    server.add_insecure_port("[::]:50051")
    server.start()
    print("gRPC Server started on port 50051")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=5)  # 5-second grace period


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
```

### 5.6 Implementing a gRPC Client (All Four Patterns)

```python
"""client.py -- Full gRPC client demonstrating all four patterns."""

import grpc
import user_pb2
import user_pb2_grpc


def run():
    # Create a channel (connection) to the server
    channel = grpc.insecure_channel("localhost:50051")

    # Create a stub (client proxy)
    stub = user_pb2_grpc.UserServiceStub(channel)

    # ─── 1. Unary RPC ───
    print("--- Unary RPC: GetUser ---")
    try:
        response = stub.GetUser(user_pb2.GetUserRequest(id=1))
        print(f"Got user: {response.user.name} ({response.user.email})")
    except grpc.RpcError as e:
        print(f"Error: {e.code()} - {e.details()}")

    # ─── 2. Server Streaming RPC ───
    print("\n--- Server Streaming: ListUsers ---")
    for user in stub.ListUsers(user_pb2.ListUsersRequest(page_size=10)):
        print(f"  Received: {user.name}")

    # ─── 3. Client Streaming RPC ───
    print("\n--- Client Streaming: BulkCreateUsers ---")

    def generate_users():
        """Generator that yields CreateUserRequest messages."""
        new_users = [
            ("Dave", "dave@example.com"),
            ("Eve", "eve@example.com"),
        ]
        for name, email in new_users:
            req = user_pb2.CreateUserRequest(
                user=user_pb2.User(name=name, email=email)
            )
            print(f"  Sending: {name}")
            yield req

    response = stub.BulkCreateUsers(generate_users())
    print(f"  Created {len(response.users)} users")

    # ─── 4. Bidirectional Streaming RPC ───
    print("\n--- Bidirectional Streaming: SyncUsers ---")

    def generate_sync_users():
        users = [
            user_pb2.User(id=1, name="alice", email="alice@example.com"),
            user_pb2.User(id=2, name="bob", email="bob@example.com"),
        ]
        for u in users:
            print(f"  Sent: {u.name}")
            yield u

    for user in stub.SyncUsers(generate_sync_users()):
        print(f"  Received back: {user.name}")  # ALICE, BOB (uppercased)

    channel.close()


if __name__ == "__main__":
    run()
```

---

## 6. Async gRPC with Python

The `grpc.aio` module provides a fully async API using Python's `asyncio`.

### 6.1 Why async gRPC?

- Better performance for I/O-bound workloads (DB queries, external API calls)
- Fits naturally with async frameworks (FastAPI, asyncpg, aiohttp)
- More efficient resource usage (no thread pool needed)

### 6.2 Async Server

```python
"""async_server.py -- Async gRPC server using grpc.aio."""

import grpc.aio
import asyncio

import user_pb2
import user_pb2_grpc


class AsyncUserServiceServicer(user_pb2_grpc.UserServiceServicer):

    async def GetUser(self, request, context):
        """Async unary RPC -- can use await."""
        # Simulate async DB call
        user_data = await self._fetch_user_from_db(request.id)

        if not user_data:
            await context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"User {request.id} not found",
            )
            # abort() raises an exception, so code below never runs

        user = user_pb2.User(**user_data)
        return user_pb2.GetUserResponse(user=user)

    async def ListUsers(self, request, context):
        """Async server streaming -- use async yield."""
        for i in range(10):
            await asyncio.sleep(0.1)  # simulate async work
            yield user_pb2.User(id=i, name=f"User {i}")

    async def BulkCreateUsers(self, request_iterator, context):
        """Async client streaming -- use async for."""
        created = []
        async for req in request_iterator:
            # Process each streamed request
            created.append(user_pb2.User(
                id=len(created) + 1,
                name=req.user.name,
            ))
        return user_pb2.ListUsersResponse(users=created)

    async def SyncUsers(self, request_iterator, context):
        """Async bidirectional streaming."""
        async for user in request_iterator:
            processed = user_pb2.User(
                id=user.id,
                name=user.name.upper(),
                email=user.email,
            )
            yield processed

    async def _fetch_user_from_db(self, user_id: int):
        """Simulate async database query."""
        await asyncio.sleep(0.01)
        db = {1: {"id": 1, "name": "Alice", "email": "alice@example.com"}}
        return db.get(user_id)


async def serve():
    server = grpc.aio.server()
    user_pb2_grpc.add_UserServiceServicer_to_server(
        AsyncUserServiceServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    await server.start()
    print("Async gRPC Server started on port 50051")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
```

### 6.3 Async Client

```python
"""async_client.py -- Async gRPC client."""

import grpc.aio
import asyncio

import user_pb2
import user_pb2_grpc


async def run():
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = user_pb2_grpc.UserServiceStub(channel)

        # Unary
        response = await stub.GetUser(user_pb2.GetUserRequest(id=1))
        print(f"User: {response.user.name}")

        # Server streaming
        async for user in stub.ListUsers(user_pb2.ListUsersRequest()):
            print(f"Streamed: {user.name}")

        # Client streaming
        async def user_generator():
            for name in ["Dave", "Eve"]:
                yield user_pb2.CreateUserRequest(
                    user=user_pb2.User(name=name, email=f"{name.lower()}@example.com")
                )

        response = await stub.BulkCreateUsers(user_generator())
        print(f"Created {len(response.users)} users")


if __name__ == "__main__":
    asyncio.run(run())
```

### 6.4 Key Differences: Sync vs Async gRPC

```
Sync (grpc)                          Async (grpc.aio)
──────────────                       ────────────────
grpc.server(ThreadPoolExecutor())    grpc.aio.server()
grpc.insecure_channel()              grpc.aio.insecure_channel()
def GetUser(self, request, ctx)      async def GetUser(self, request, ctx)
yield response                       yield response (in async generator)
for req in request_iterator          async for req in request_iterator
stub.GetUser(request)                await stub.GetUser(request)
context.set_code() + return          await context.abort() (raises exception)
```

---

## 7. gRPC Error Handling

### 7.1 gRPC Status Codes

```
Code                    HTTP Equiv.   Meaning
────                    ───────────   ───────
OK                      200           Success
CANCELLED               499           Client cancelled
INVALID_ARGUMENT        400           Bad request (validation error)
NOT_FOUND               404           Resource not found
ALREADY_EXISTS          409           Conflict (duplicate)
PERMISSION_DENIED       403           Forbidden
UNAUTHENTICATED         401           Not authenticated
RESOURCE_EXHAUSTED      429           Rate limited / quota exceeded
FAILED_PRECONDITION     400           State precondition failed
ABORTED                 409           Concurrency conflict
OUT_OF_RANGE            400           Value out of valid range
UNIMPLEMENTED           501           Method not implemented
INTERNAL                500           Internal server error
UNAVAILABLE             503           Service temporarily unavailable
DATA_LOSS               500           Unrecoverable data loss
DEADLINE_EXCEEDED       504           Timeout
```

### 7.2 Server-Side Error Handling

```python
class UserServiceServicer(user_pb2_grpc.UserServiceServicer):

    def GetUser(self, request, context):
        # ─── Validation ───
        if request.id <= 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("User ID must be a positive integer")
            return user_pb2.GetUserResponse()

        # ─── Not found ───
        user = db.get(request.id)
        if not user:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"User {request.id} not found")
            return user_pb2.GetUserResponse()

        # ─── Internal error ───
        try:
            result = process_user(user)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Processing failed: {str(e)}")
            return user_pb2.GetUserResponse()

        return user_pb2.GetUserResponse(user=result)
```

### 7.3 Async Error Handling with `context.abort()`

```python
class AsyncUserServiceServicer(user_pb2_grpc.UserServiceServicer):

    async def GetUser(self, request, context):
        if request.id <= 0:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "User ID must be a positive integer",
            )
            # abort() raises AbortError, so no return needed

        user = await db.get(request.id)
        if not user:
            await context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"User {request.id} not found",
            )

        return user_pb2.GetUserResponse(user=user)
```

### 7.4 Rich Error Details (google.rpc.Status)

```python
from grpc_status import rpc_status
from google.protobuf import any_pb2
from google.rpc import status_pb2, error_details_pb2

def GetUser(self, request, context):
    if request.id <= 0:
        # Create rich error with field violations
        bad_request = error_details_pb2.BadRequest()
        violation = bad_request.field_violations.add()
        violation.field = "id"
        violation.description = "User ID must be a positive integer"

        # Pack into Any
        detail = any_pb2.Any()
        detail.Pack(bad_request)

        # Create status with details
        rich_status = status_pb2.Status(
            code=grpc.StatusCode.INVALID_ARGUMENT.value[0],
            message="Invalid request",
            details=[detail],
        )

        context.abort_with_status(rpc_status.to_status(rich_status))
```

### 7.5 Client-Side Error Handling

```python
import grpc

try:
    response = stub.GetUser(user_pb2.GetUserRequest(id=-1))
except grpc.RpcError as e:
    print(f"gRPC Error:")
    print(f"  Code:    {e.code()}")             # StatusCode.INVALID_ARGUMENT
    print(f"  Details: {e.details()}")           # "User ID must be positive"

    # Handle specific error codes
    if e.code() == grpc.StatusCode.NOT_FOUND:
        print("User does not exist")
    elif e.code() == grpc.StatusCode.UNAVAILABLE:
        print("Server is down, retrying...")
    elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
        print("Request timed out")
    elif e.code() == grpc.StatusCode.UNAUTHENTICATED:
        print("Need to refresh token")
```

---

## 8. gRPC Interceptors (Middleware)

Interceptors are gRPC's equivalent of middleware. They intercept RPC calls on the client or server side for cross-cutting concerns like logging, authentication, metrics, and retries.

### 8.1 Server Interceptors

```python
import grpc
import time
import logging

logger = logging.getLogger(__name__)


class LoggingInterceptor(grpc.ServerInterceptor):
    """Log every RPC call with duration."""

    def intercept_service(self, continuation, handler_call_details):
        method = handler_call_details.method  # e.g., "/user.v1.UserService/GetUser"
        start = time.perf_counter()
        logger.info(f"RPC Start: {method}")

        # Call the actual handler
        response = continuation(handler_call_details)

        duration = time.perf_counter() - start
        logger.info(f"RPC End: {method} ({duration:.3f}s)")

        return response


class AuthInterceptor(grpc.ServerInterceptor):
    """Validate JWT tokens from metadata."""

    def __init__(self, valid_tokens: set):
        self._valid_tokens = valid_tokens

    def intercept_service(self, continuation, handler_call_details):
        # Extract metadata
        metadata = dict(handler_call_details.invocation_metadata or [])
        token = metadata.get("authorization", "")

        if not token.startswith("Bearer "):
            # Return an error handler that denies access
            return self._unauthenticated_handler()

        bearer_token = token[7:]
        if bearer_token not in self._valid_tokens:
            return self._unauthenticated_handler()

        # Token valid -- proceed to actual handler
        return continuation(handler_call_details)

    def _unauthenticated_handler(self):
        def deny(request, context):
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details("Invalid or missing token")
            raise grpc.RpcError()
        return grpc.unary_unary_rpc_method_handler(deny)


# Register interceptors with server
server = grpc.server(
    futures.ThreadPoolExecutor(max_workers=10),
    interceptors=[
        LoggingInterceptor(),
        AuthInterceptor(valid_tokens={"secret-token-123"}),
    ],
)
```

### 8.2 Client Interceptors

```python
class RetryInterceptor(
    grpc.UnaryUnaryClientInterceptor,
):
    """Retry failed unary calls."""

    def __init__(self, max_retries=3, retry_codes=None):
        self._max_retries = max_retries
        self._retry_codes = retry_codes or {
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.DEADLINE_EXCEEDED,
        }

    def intercept_unary_unary(self, continuation, client_call_details, request):
        for attempt in range(self._max_retries):
            response = continuation(client_call_details, request)
            # Check if it's an error that should be retried
            try:
                result = response.result()
                return response
            except grpc.RpcError as e:
                if e.code() not in self._retry_codes:
                    raise
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # exponential backoff


class AuthClientInterceptor(grpc.UnaryUnaryClientInterceptor):
    """Attach JWT token to every request."""

    def __init__(self, token: str):
        self._token = token

    def intercept_unary_unary(self, continuation, client_call_details, request):
        # Add metadata to the call
        metadata = list(client_call_details.metadata or [])
        metadata.append(("authorization", f"Bearer {self._token}"))

        new_details = client_call_details._replace(metadata=metadata)
        return continuation(new_details, request)


# Use interceptors with channel
channel = grpc.intercept_channel(
    grpc.insecure_channel("localhost:50051"),
    AuthClientInterceptor("secret-token-123"),
    RetryInterceptor(max_retries=3),
)
stub = user_pb2_grpc.UserServiceStub(channel)
```

### 8.3 Interceptor Execution Order

```
Client Request
    │
    ▼
┌─────────────────┐
│ Auth Interceptor │   ← adds token to metadata
└────────┬────────┘
         ▼
┌─────────────────┐
│ Retry Interceptor│   ← wraps call with retry logic
└────────┬────────┘
         ▼
    ── network ──
         ▼
┌─────────────────┐
│ Logging Intercept│   ← logs method + duration
└────────┬────────┘
         ▼
┌─────────────────┐
│ Auth Interceptor │   ← validates token
└────────┬────────┘
         ▼
┌─────────────────┐
│  Actual Handler  │   ← your business logic
└─────────────────┘
```

---

## 9. gRPC Advanced Topics

### 9.1 Deadlines and Timeouts

Deadlines propagate across service calls (critical for microservices):

```python
# Client: set a 5-second deadline
try:
    response = stub.GetUser(
        user_pb2.GetUserRequest(id=1),
        timeout=5.0,  # seconds
    )
except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
        print("Request timed out after 5 seconds")

# Server: check remaining time
def GetUser(self, request, context):
    remaining = context.time_remaining()  # seconds until deadline
    if remaining < 0.5:
        context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
        context.set_details("Not enough time remaining")
        return user_pb2.GetUserResponse()
```

```
Deadline propagation across services:

Client (timeout=5s) ──► Service A (4.8s left) ──► Service B (4.5s left) ──► DB
   If Service B is slow, ALL upstream calls fail with DEADLINE_EXCEEDED.
```

### 9.2 Metadata (Headers)

```python
# Client: send metadata
metadata = [
    ("x-request-id", "abc-123"),
    ("x-client-version", "1.0.0"),
    ("authorization", "Bearer my-jwt-token"),
]
response = stub.GetUser(
    user_pb2.GetUserRequest(id=1),
    metadata=metadata,
)

# Server: read metadata
def GetUser(self, request, context):
    metadata = dict(context.invocation_metadata())
    request_id = metadata.get("x-request-id", "unknown")
    token = metadata.get("authorization", "")

    # Send response metadata (trailing headers)
    context.set_trailing_metadata([
        ("x-processing-time", "0.023s"),
    ])
    return user_pb2.GetUserResponse(user=user)

# Client: read response metadata
call = stub.GetUser.with_call(user_pb2.GetUserRequest(id=1))
response, call_obj = call
trailing_metadata = dict(call_obj.trailing_metadata())
print(trailing_metadata.get("x-processing-time"))
```

### 9.3 Compression

```python
# Client-side compression
response = stub.GetUser(
    user_pb2.GetUserRequest(id=1),
    compression=grpc.Compression.Gzip,
)

# Server-side: enable compression globally
server = grpc.server(
    futures.ThreadPoolExecutor(max_workers=10),
    compression=grpc.Compression.Gzip,
)
```

### 9.4 Load Balancing

```python
# Client-side load balancing (round-robin)
channel = grpc.insecure_channel(
    "dns:///my-service:50051",
    options=[
        ("grpc.lb_policy_name", "round_robin"),
    ],
)

# For Kubernetes / service mesh:
#   - Use headless service (returns all pod IPs)
#   - Client resolves DNS and balances across pods
#   - Or use a service mesh like Istio/Linkerd (transparent to app code)
```

```
Load Balancing Strategies:
──────────────────────────

1. Client-side (built-in):
   Client ──► DNS ──► [Pod1, Pod2, Pod3]
   Client round-robins between pods

2. Proxy (L7 load balancer):
   Client ──► Envoy/nginx ──► [Pod1, Pod2, Pod3]
   Proxy distributes requests

3. Service Mesh (Istio/Linkerd):
   Client ──► Sidecar ──► [Pod1, Pod2, Pod3]
   Transparent to application
```

### 9.5 Health Checking

```python
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

# Add health service to server
health_servicer = health.HealthServicer()
health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

# Set service status
health_servicer.set("user.v1.UserService", health_pb2.HealthCheckResponse.SERVING)

# Mark as not serving (during shutdown)
health_servicer.set("user.v1.UserService", health_pb2.HealthCheckResponse.NOT_SERVING)
```

```bash
# Check health with grpcurl
grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check
```

### 9.6 Reflection (for Debugging)

Reflection lets clients discover available services at runtime (like Swagger for gRPC).

```python
from grpc_reflection.v1alpha import reflection

# Enable reflection on the server
SERVICE_NAMES = (
    user_pb2.DESCRIPTOR.services_by_name["UserService"].full_name,
    reflection.SERVICE_NAME,  # reflection service itself
)
reflection.enable_server_reflection(SERVICE_NAMES, server)
```

```bash
# Now you can explore services with grpcurl (without .proto files!)
grpcurl -plaintext localhost:50051 list
# Output:
#   grpc.health.v1.Health
#   grpc.reflection.v1alpha.ServerReflection
#   user.v1.UserService

grpcurl -plaintext localhost:50051 describe user.v1.UserService
# Shows all methods

grpcurl -plaintext -d '{"id": 1}' localhost:50051 user.v1.UserService/GetUser
# Actually calls the method
```

### 9.7 TLS/SSL Configuration

```python
# ─── Server with TLS ───
with open("server.key", "rb") as f:
    private_key = f.read()
with open("server.crt", "rb") as f:
    certificate_chain = f.read()

server_credentials = grpc.ssl_server_credentials(
    [(private_key, certificate_chain)]
)
server.add_secure_port("[::]:50051", server_credentials)

# ─── Client with TLS ───
with open("ca.crt", "rb") as f:
    trusted_certs = f.read()

channel_credentials = grpc.ssl_channel_credentials(
    root_certificates=trusted_certs
)
channel = grpc.secure_channel("my-server:50051", channel_credentials)
```

### 9.8 Connection Pooling

```python
# gRPC channels are already connection-pooled internally.
# A single channel multiplexes many RPC calls over one HTTP/2 connection.
# You typically only need ONE channel per target server.

# For high-throughput, you can create multiple channels:
channels = [
    grpc.insecure_channel("server:50051")
    for _ in range(4)  # 4 channels = 4 HTTP/2 connections
]

# Round-robin across channels
import itertools
channel_pool = itertools.cycle(channels)

def get_stub():
    return user_pb2_grpc.UserServiceStub(next(channel_pool))
```

---

## 10. gRPC + FastAPI Integration

### 10.1 Why Run Both?

```
                    ┌─────────────────────────┐
  Browser/Mobile ──►│   FastAPI (REST/HTTP)    │ Port 8000
                    │   - Public APIs          │
                    │   - OpenAPI docs          │
                    │   - Browser-friendly      │
                    └────────────┬──────────────┘
                                 │ shared service layer
                    ┌────────────┴──────────────┐
  Internal μsvc ───►│   gRPC Server             │ Port 50051
                    │   - Internal APIs          │
                    │   - High-performance       │
                    │   - Streaming              │
                    └─────────────────────────────┘
```

### 10.2 Shared Service Layer (Clean Architecture)

```python
"""service.py -- Shared business logic (used by both REST and gRPC)."""

from dataclasses import dataclass


@dataclass
class UserDTO:
    id: int
    name: str
    email: str
    roles: list[str]


class UserService:
    """Business logic shared between REST and gRPC layers."""

    def __init__(self, db):
        self.db = db

    async def get_user(self, user_id: int) -> UserDTO | None:
        row = await self.db.fetchone("SELECT * FROM users WHERE id = ?", user_id)
        if not row:
            return None
        return UserDTO(**row)

    async def create_user(self, name: str, email: str) -> UserDTO:
        user_id = await self.db.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)", name, email
        )
        return UserDTO(id=user_id, name=name, email=email, roles=[])

    async def list_users(self, page_size: int = 10, offset: int = 0):
        rows = await self.db.fetchall(
            "SELECT * FROM users LIMIT ? OFFSET ?", page_size, offset
        )
        return [UserDTO(**row) for row in rows]
```

```python
"""rest_api.py -- FastAPI layer (REST)."""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

app = FastAPI()

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    roles: list[str]

class UserCreate(BaseModel):
    name: str
    email: str

async def get_user_service():
    return UserService(db=get_database())

@app.get("/api/v1/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, svc: UserService = Depends(get_user_service)):
    user = await svc.get_user(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    return user
```

```python
"""grpc_api.py -- gRPC layer (uses same UserService)."""

import grpc.aio
import user_pb2
import user_pb2_grpc

class GrpcUserServicer(user_pb2_grpc.UserServiceServicer):
    def __init__(self, user_service: UserService):
        self.svc = user_service

    async def GetUser(self, request, context):
        user = await self.svc.get_user(request.id)
        if not user:
            await context.abort(grpc.StatusCode.NOT_FOUND, "User not found")
        return user_pb2.GetUserResponse(
            user=user_pb2.User(
                id=user.id, name=user.name, email=user.email, roles=user.roles
            )
        )
```

### 10.3 Running Both Servers Together

```python
"""main.py -- Run FastAPI and gRPC in the same process."""

import asyncio
import grpc.aio
import uvicorn

from rest_api import app
from grpc_api import GrpcUserServicer
import user_pb2_grpc


async def start_grpc_server(user_service):
    server = grpc.aio.server()
    user_pb2_grpc.add_UserServiceServicer_to_server(
        GrpcUserServicer(user_service), server
    )
    server.add_insecure_port("[::]:50051")
    await server.start()
    print("gRPC server started on port 50051")
    return server


async def main():
    user_service = UserService(db=get_database())

    # Start gRPC server
    grpc_server = await start_grpc_server(user_service)

    # Start FastAPI with uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    uvicorn_server = uvicorn.Server(config)

    # Run both concurrently
    await asyncio.gather(
        grpc_server.wait_for_termination(),
        uvicorn_server.serve(),
    )


if __name__ == "__main__":
    asyncio.run(main())
```

### 10.4 gRPC-Gateway Pattern (REST Proxy)

For exposing a gRPC service as REST without writing separate code. Typically done with Envoy or grpc-gateway (Go).

```
Browser ──REST──► Envoy Proxy ──gRPC──► gRPC Server
                  (transcoding)

Envoy config:
  - Maps GET /api/users/42  →  UserService.GetUser({id: 42})
  - Maps POST /api/users    →  UserService.CreateUser({...})
  - Translates JSON ↔ Protobuf automatically
```

---

## 11. API Versioning & Authentication

### 11.1 REST Versioning

```
/api/v1/users    ← version in URL path (most common)
/api/v2/users    ← new version with breaking changes
```

### 11.2 gRPC Versioning (Package-Based)

```protobuf
// Version 1
package user.v1;
service UserService { ... }

// Version 2 (coexists with v1)
package user.v2;
service UserService { ... }  // can have different methods/messages
```

```
Both versions run on the same server:
  user.v1.UserService/GetUser
  user.v2.UserService/GetUser   ← new version, same method name
```

**Backward-compatible changes** (no new version needed):
- Adding new fields to messages (use new field numbers)
- Adding new methods to a service
- Adding new services

**Breaking changes** (new version required):
- Removing or renaming fields
- Changing field types
- Changing method signatures
- Reordering or reusing field numbers

### 11.3 JWT Authentication with gRPC

```python
# ─── Client: attach JWT to every call ───
class JWTClientInterceptor(grpc.UnaryUnaryClientInterceptor):
    def __init__(self, token: str):
        self._token = token

    def intercept_unary_unary(self, continuation, client_call_details, request):
        metadata = list(client_call_details.metadata or [])
        metadata.append(("authorization", f"Bearer {self._token}"))
        new_details = client_call_details._replace(metadata=metadata)
        return continuation(new_details, request)

# Usage
channel = grpc.intercept_channel(
    grpc.insecure_channel("localhost:50051"),
    JWTClientInterceptor(token="eyJhbGciOiJI..."),
)

# ─── Server: validate JWT in interceptor ───
import jwt

class JWTServerInterceptor(grpc.ServerInterceptor):
    def __init__(self, secret_key: str):
        self._secret = secret_key

    def intercept_service(self, continuation, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata or [])
        token = metadata.get("authorization", "")

        if not token.startswith("Bearer "):
            return self._deny("Missing Bearer token")

        try:
            payload = jwt.decode(token[7:], self._secret, algorithms=["HS256"])
            # Could attach user info to context here
        except jwt.ExpiredSignatureError:
            return self._deny("Token expired")
        except jwt.InvalidTokenError:
            return self._deny("Invalid token")

        return continuation(handler_call_details)

    def _deny(self, message):
        def handler(request, context):
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details(message)
            return None
        return grpc.unary_unary_rpc_method_handler(handler)
```

### 11.4 API Keys in gRPC Metadata

```python
# Client
metadata = [("x-api-key", "sk-abc123def456")]
response = stub.GetUser(user_pb2.GetUserRequest(id=1), metadata=metadata)

# Server (in interceptor or handler)
def GetUser(self, request, context):
    metadata = dict(context.invocation_metadata())
    api_key = metadata.get("x-api-key")
    if not api_key or not validate_api_key(api_key):
        context.set_code(grpc.StatusCode.UNAUTHENTICATED)
        context.set_details("Invalid API key")
        return user_pb2.GetUserResponse()
```

### 11.5 Rate Limiting in gRPC

```python
import time
from collections import defaultdict

class RateLimitInterceptor(grpc.ServerInterceptor):
    """Simple token-bucket rate limiter."""

    def __init__(self, requests_per_second: int = 10):
        self._rate = requests_per_second
        self._tokens = defaultdict(lambda: requests_per_second)
        self._last_refill = defaultdict(time.time)

    def intercept_service(self, continuation, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata or [])
        client_id = metadata.get("x-api-key", "anonymous")

        # Refill tokens
        now = time.time()
        elapsed = now - self._last_refill[client_id]
        self._tokens[client_id] = min(
            self._rate, self._tokens[client_id] + elapsed * self._rate
        )
        self._last_refill[client_id] = now

        # Check tokens
        if self._tokens[client_id] < 1:
            def deny(request, context):
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                context.set_details("Rate limit exceeded")
                return None
            return grpc.unary_unary_rpc_method_handler(deny)

        self._tokens[client_id] -= 1
        return continuation(handler_call_details)
```

---

## 12. Testing gRPC

### 12.1 Unit Testing Servicers

```python
"""test_user_service.py -- Unit tests for gRPC servicer."""

import unittest
from unittest.mock import MagicMock
import grpc

import user_pb2
import user_pb2_grpc
from server import UserServiceServicer


class TestUserServiceServicer(unittest.TestCase):

    def setUp(self):
        self.servicer = UserServiceServicer()

    def test_get_user_success(self):
        # Arrange
        request = user_pb2.GetUserRequest(id=1)
        context = MagicMock()

        # Act
        response = self.servicer.GetUser(request, context)

        # Assert
        self.assertEqual(response.user.name, "Alice")
        self.assertEqual(response.user.email, "alice@example.com")
        context.set_code.assert_not_called()

    def test_get_user_not_found(self):
        request = user_pb2.GetUserRequest(id=999)
        context = MagicMock()

        response = self.servicer.GetUser(request, context)

        context.set_code.assert_called_once_with(grpc.StatusCode.NOT_FOUND)

    def test_list_users_streaming(self):
        request = user_pb2.ListUsersRequest()
        context = MagicMock()

        # Server streaming returns a generator
        users = list(self.servicer.ListUsers(request, context))

        self.assertGreater(len(users), 0)
        self.assertIsInstance(users[0], user_pb2.User)
```

### 12.2 Integration Testing with a Real gRPC Server

```python
"""test_integration.py -- Spin up a real server for testing."""

import unittest
import grpc
from concurrent import futures

import user_pb2
import user_pb2_grpc
from server import UserServiceServicer


class TestUserServiceIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Start a gRPC server for tests."""
        cls.server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        user_pb2_grpc.add_UserServiceServicer_to_server(
            UserServiceServicer(), cls.server
        )
        cls.port = cls.server.add_insecure_port("[::]:0")  # random port
        cls.server.start()

        cls.channel = grpc.insecure_channel(f"localhost:{cls.port}")
        cls.stub = user_pb2_grpc.UserServiceStub(cls.channel)

    @classmethod
    def tearDownClass(cls):
        cls.channel.close()
        cls.server.stop(grace=0)

    def test_get_user(self):
        response = self.stub.GetUser(user_pb2.GetUserRequest(id=1))
        self.assertEqual(response.user.name, "Alice")

    def test_get_user_not_found(self):
        with self.assertRaises(grpc.RpcError) as ctx:
            self.stub.GetUser(user_pb2.GetUserRequest(id=999))
        self.assertEqual(ctx.exception.code(), grpc.StatusCode.NOT_FOUND)

    def test_list_users_streaming(self):
        users = list(self.stub.ListUsers(user_pb2.ListUsersRequest()))
        self.assertGreater(len(users), 0)
```

### 12.3 Async Integration Tests (pytest)

```python
"""test_async.py -- Async gRPC tests with pytest-asyncio."""

import pytest
import grpc.aio

import user_pb2
import user_pb2_grpc
from async_server import AsyncUserServiceServicer


@pytest.fixture
async def grpc_channel():
    """Fixture: start async server and provide a channel."""
    server = grpc.aio.server()
    user_pb2_grpc.add_UserServiceServicer_to_server(
        AsyncUserServiceServicer(), server
    )
    port = server.add_insecure_port("[::]:0")
    await server.start()

    async with grpc.aio.insecure_channel(f"localhost:{port}") as channel:
        yield channel

    await server.stop(grace=0)


@pytest.fixture
async def stub(grpc_channel):
    return user_pb2_grpc.UserServiceStub(grpc_channel)


@pytest.mark.asyncio
async def test_get_user(stub):
    response = await stub.GetUser(user_pb2.GetUserRequest(id=1))
    assert response.user.name == "Alice"


@pytest.mark.asyncio
async def test_stream_users(stub):
    users = []
    async for user in stub.ListUsers(user_pb2.ListUsersRequest()):
        users.append(user)
    assert len(users) > 0
```

### 12.4 Manual Testing with grpcurl

```bash
# Install grpcurl
# macOS: brew install grpcurl
# Linux: go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest

# List services (requires reflection enabled)
grpcurl -plaintext localhost:50051 list

# Describe a service
grpcurl -plaintext localhost:50051 describe user.v1.UserService

# Describe a message type
grpcurl -plaintext localhost:50051 describe user.v1.User

# Call a unary method
grpcurl -plaintext \
    -d '{"id": 1}' \
    localhost:50051 user.v1.UserService/GetUser

# Call with metadata (headers)
grpcurl -plaintext \
    -H 'authorization: Bearer my-jwt-token' \
    -d '{"id": 1}' \
    localhost:50051 user.v1.UserService/GetUser

# Use .proto file instead of reflection
grpcurl -plaintext \
    -import-path ./protos \
    -proto user.proto \
    -d '{"id": 1}' \
    localhost:50051 user.v1.UserService/GetUser
```

---

## 13. Q&A Section

### Q1. What is gRPC and how does it differ from REST?

**Answer**: gRPC (Google Remote Procedure Call) is a high-performance RPC framework that uses Protocol Buffers for serialization and HTTP/2 for transport. Unlike REST, which is resource-oriented (you manipulate resources via URLs and HTTP verbs), gRPC is action-oriented (you call remote methods like local functions).

Key differences:
- **Data format**: REST uses JSON (text); gRPC uses Protobuf (binary, ~10x smaller)
- **Protocol**: REST typically uses HTTP/1.1; gRPC requires HTTP/2 (multiplexing, streaming)
- **Contract**: REST can work without a schema; gRPC requires a `.proto` file (strongly typed)
- **Code generation**: REST code gen is optional; gRPC generates client/server stubs automatically
- **Streaming**: REST has limited streaming (SSE, WebSocket); gRPC has native bidirectional streaming
- **Performance**: gRPC is typically 7-10x faster due to binary serialization and HTTP/2

---

### Q2. What are Protocol Buffers and how do they work?

**Answer**: Protocol Buffers (Protobuf) are Google's language-neutral, platform-neutral, extensible mechanism for serializing structured data. You define data structures and services in `.proto` files, then use the `protoc` compiler to generate code for your target language.

```protobuf
syntax = "proto3";

message Person {
    string name = 1;    // field number 1
    int32 age = 2;      // field number 2
}
```

The field numbers (1, 2) are used in the binary encoding, not the field names. This means you can rename fields without breaking compatibility. Protobuf is more compact and faster than JSON because it uses a binary wire format (varint encoding for integers, length-delimited for strings).

---

### Q3. Explain the 4 types of gRPC communication patterns.

**Answer**:

1. **Unary RPC**: Client sends one request, server sends one response. Like a normal function call or REST request. Example: `GetUser(id) -> User`.

2. **Server streaming RPC**: Client sends one request, server responds with a stream of messages. Example: `ListUsers(filter) -> stream of User`. Useful for large result sets or real-time feeds.

3. **Client streaming RPC**: Client sends a stream of messages, server responds with a single message after receiving all. Example: `BulkUpload(stream of Records) -> Summary`. Useful for file uploads or batch operations.

4. **Bidirectional streaming RPC**: Both client and server send streams of messages independently. Neither has to wait for the other. Example: chat, real-time collaboration, live ML inference with feedback.

---

### Q4. How does HTTP/2 benefit gRPC?

**Answer**: HTTP/2 provides several critical features for gRPC:

- **Multiplexing**: Multiple RPC calls share a single TCP connection concurrently, eliminating head-of-line blocking at the HTTP level
- **Binary framing**: Data is sent in binary frames, which is more efficient to parse than HTTP/1.1 text
- **Header compression (HPACK)**: Repeated headers are compressed, reducing overhead
- **Bidirectional streaming**: Both sides can send data simultaneously over the same connection
- **Server push**: Server can initiate sending data without the client requesting it
- **Persistent connections**: One connection serves many requests, reducing TCP handshake overhead

---

### Q5. What is bidirectional streaming and when would you use it?

**Answer**: Bidirectional streaming is a gRPC pattern where both client and server can send messages independently at any time on the same connection. The two streams are completely independent -- the client does not have to wait for a server response before sending the next message, and vice versa.

**Use cases**:
- **Chat applications**: messages flow freely in both directions
- **Live ML inference with feedback**: send data, receive predictions, send corrections in real-time
- **Collaborative editing**: multiple users pushing and receiving changes
- **Game state sync**: real-time position/state updates in both directions
- **IoT sensor monitoring**: devices stream data up, server streams commands down

```protobuf
rpc Chat(stream ChatMessage) returns (stream ChatMessage);
```

---

### Q6. How do you handle errors in gRPC?

**Answer**: gRPC uses its own set of status codes (16 codes total, similar to but different from HTTP status codes). On the server, you set the error code and message on the `context` object. On the client, you catch `grpc.RpcError`.

**Server side** (sync):
```python
context.set_code(grpc.StatusCode.NOT_FOUND)
context.set_details("User not found")
return empty_response
```

**Server side** (async):
```python
await context.abort(grpc.StatusCode.NOT_FOUND, "User not found")
# abort raises an exception, no return needed
```

**Client side**:
```python
try:
    response = stub.GetUser(request)
except grpc.RpcError as e:
    print(e.code())     # grpc.StatusCode.NOT_FOUND
    print(e.details())  # "User not found"
```

For rich error details, you can use `google.rpc.Status` with packed `Any` details (field violations, debug info, etc.).

---

### Q7. What are gRPC interceptors and how do they work?

**Answer**: Interceptors are gRPC's equivalent of middleware. They intercept RPC calls on the client or server side before the actual handler runs. They are used for cross-cutting concerns.

**Common uses**: authentication, logging, metrics, rate limiting, tracing, retries, error handling.

**Server interceptor**: implements `grpc.ServerInterceptor` with `intercept_service()`. It receives `handler_call_details` (method name, metadata) and a `continuation` function. You either call `continuation` to proceed or return your own handler to short-circuit.

**Client interceptor**: implements e.g. `grpc.UnaryUnaryClientInterceptor` with `intercept_unary_unary()`. It wraps the outgoing call, letting you modify metadata, add retries, etc.

Interceptors are registered when creating the server or channel, and they execute in order.

---

### Q8. How does gRPC achieve better performance than REST?

**Answer**: Several factors contribute:

1. **Binary serialization (Protobuf)**: Messages are encoded in a compact binary format (vs. text JSON). Protobuf is ~5-10x smaller and ~20-100x faster to parse than JSON.

2. **HTTP/2 multiplexing**: Multiple RPCs share one TCP connection. REST over HTTP/1.1 requires separate connections or suffers from head-of-line blocking.

3. **Header compression**: HTTP/2 HPACK compresses headers. REST repeats full headers on every request.

4. **Persistent connections**: gRPC keeps connections open and reuses them. REST often creates new connections.

5. **No text parsing**: Binary framing eliminates the overhead of parsing text-based protocols.

6. **Schema-based**: The compiler can generate optimized serialization/deserialization code.

Typical benchmarks show gRPC is 7-10x faster than REST+JSON for equivalent workloads.

---

### Q9. What is the .proto file and how is code generated from it?

**Answer**: A `.proto` file is the Interface Definition Language (IDL) file for gRPC. It defines:
- **Messages**: data structures (like classes/structs)
- **Services**: collections of RPC methods
- **Enums**: enumerated types
- **Imports**: references to other `.proto` files

Code generation uses the `protoc` compiler with language-specific plugins:

```bash
python -m grpc_tools.protoc \
    -I./protos \
    --python_out=./generated \       # generates _pb2.py (messages)
    --grpc_python_out=./generated \  # generates _pb2_grpc.py (services)
    ./protos/user.proto
```

The generated `_pb2.py` contains message classes (User, GetUserRequest). The generated `_pb2_grpc.py` contains the service stub (client) and servicer base class (server). You never edit generated code -- you subclass the servicer and use the stub as-is.

---

### Q10. How do you implement authentication in gRPC?

**Answer**: Authentication in gRPC is typically done via **metadata** (similar to HTTP headers) and **interceptors** (middleware):

1. **JWT tokens**: Client sends `authorization: Bearer <token>` in metadata. Server interceptor validates the JWT and extracts claims.

2. **API keys**: Client sends `x-api-key: <key>` in metadata. Server interceptor validates the key.

3. **Mutual TLS (mTLS)**: Both client and server present certificates. The transport layer handles authentication.

4. **Google token-based**: Using `grpc.composite_channel_credentials` with Google OAuth2 tokens (for Google Cloud services).

The interceptor pattern is preferred because it separates auth logic from business logic. The interceptor checks credentials and either allows the call to proceed or aborts with `UNAUTHENTICATED`.

---

### Q11. What are gRPC deadlines and how do they work?

**Answer**: Deadlines (timeouts) in gRPC specify the maximum time a client is willing to wait for an RPC to complete. Unlike REST timeouts, gRPC deadlines **propagate across service calls**.

```python
# Client sets a 5-second deadline
response = stub.GetUser(request, timeout=5.0)
```

If Service A calls Service B, the remaining deadline is automatically passed along. If Service B calls Service C, the deadline shrinks further. If any service takes too long, all upstream calls fail with `DEADLINE_EXCEEDED`.

```
Client (5s) -> A (4.8s left) -> B (4.5s left) -> C
If C takes 5s, everything fails.
```

On the server, you can check remaining time with `context.time_remaining()` and bail out early if there is not enough time for your operation.

---

### Q12. How do you version gRPC APIs?

**Answer**: gRPC uses **package-based versioning**:

```protobuf
package user.v1;   // version 1
package user.v2;   // version 2
```

Both versions can coexist on the same server. Clients connect to whichever version they support. Fully qualified method names include the package: `user.v1.UserService/GetUser` vs `user.v2.UserService/GetUser`.

**Backward-compatible changes** (no new version needed): adding new fields (with new field numbers), adding new methods, adding new services.

**Breaking changes** (need new version): removing fields, changing field types incompatibly, changing method signatures, reusing field numbers.

---

### Q13. When would you choose REST over gRPC?

**Answer**:
- **Public APIs**: REST is universally understood; third-party developers expect it
- **Browser clients**: Browsers natively speak HTTP/REST but need gRPC-Web proxy for gRPC
- **Simple CRUD**: For basic resource operations, REST is simpler and well-understood
- **HTTP caching needed**: REST leverages standard HTTP caching (ETags, Cache-Control); gRPC has no built-in HTTP caching
- **Debugging ease**: JSON is human-readable; Protobuf binary is not
- **Team familiarity**: If the team does not know gRPC, the learning curve may not be worth it for simple use cases
- **Rapid prototyping**: REST is faster to set up (no `.proto` compilation step)

---

### Q14. How do you test gRPC services?

**Answer**: Three levels:

1. **Unit tests**: Create the servicer directly, pass a mock `context`, and call methods. No network involved. Fast but does not test serialization.

2. **Integration tests**: Start a real gRPC server on a random port (`server.add_insecure_port("[::]:0")`), create a real channel and stub, make actual RPC calls. Tests the full stack including serialization.

3. **Manual testing with grpcurl**: A command-line tool that talks gRPC (like `curl` for REST). Requires server reflection or `.proto` files. Great for ad-hoc testing and debugging.

```bash
grpcurl -plaintext -d '{"id": 1}' localhost:50051 user.v1.UserService/GetUser
```

---

### Q15. What is gRPC reflection and why is it useful?

**Answer**: gRPC reflection is a server-side feature that lets clients discover available services, methods, and message types at runtime -- without having the `.proto` files. It is like Swagger/OpenAPI for gRPC.

You enable it by registering the reflection service on your server:
```python
from grpc_reflection.v1alpha import reflection
reflection.enable_server_reflection(SERVICE_NAMES, server)
```

Then tools like `grpcurl` can query the server to list services, describe methods, and even call them without any `.proto` files. This is useful for debugging, monitoring dashboards, and service discovery.

---

### Q16. How do you implement streaming in Python gRPC?

**Answer**:

**Server streaming**: The server method `yields` messages one at a time:
```python
def ListUsers(self, request, context):
    for user_data in database:
        yield user_pb2.User(id=user_data.id, name=user_data.name)
```

**Client streaming**: The server method receives a `request_iterator` and iterates over it:
```python
def BulkCreate(self, request_iterator, context):
    for req in request_iterator:
        process(req)
    return summary
```

**Bidirectional**: Combines both -- method receives an iterator and yields responses:
```python
def Chat(self, request_iterator, context):
    for message in request_iterator:
        yield process(message)
```

On the async side, you use `async for` and `yield` in `async def` methods.

---

### Q17. What is gRPC-Web and why is it needed?

**Answer**: Browsers cannot make native gRPC calls because:
1. Browsers do not expose HTTP/2 framing to JavaScript
2. Browsers do not support HTTP/2 trailers (which gRPC uses for status codes)

gRPC-Web is a protocol that translates gRPC calls into a browser-compatible format. It requires a proxy (typically Envoy) between the browser and the gRPC server:

```
Browser (gRPC-Web) ──► Envoy Proxy ──► gRPC Server
   (HTTP/1.1 or 2)     (translates)    (native gRPC)
```

gRPC-Web supports unary and server streaming only (no client or bidirectional streaming due to browser limitations).

---

### Q18. How do you handle backward compatibility in Protobuf?

**Answer**: Rules for backward-compatible changes:

**Safe**:
- Add new fields with new field numbers
- Remove fields (but `reserve` the number and name)
- Rename fields (wire format uses numbers, not names)
- Add new services or methods

**Unsafe (breaking)**:
- Reuse a field number that was previously used
- Change a field's type (some exceptions: int32 -> int64 is safe)
- Change a field from `repeated` to singular or vice versa
- Remove a field without reserving the number

```protobuf
message User {
    reserved 3, 8;               // never reuse these numbers
    reserved "old_field_name";   // document what was removed
}
```

Field numbers 1-15 use 1 byte on the wire; 16-2047 use 2 bytes. Use 1-15 for frequently set fields.

---

### Q19. What is the difference between grpc and grpc.aio in Python?

**Answer**:

| Aspect               | `grpc` (sync)                      | `grpc.aio` (async)                   |
|-----------------------|------------------------------------|---------------------------------------|
| Server creation       | `grpc.server(ThreadPoolExecutor)`  | `grpc.aio.server()`                   |
| Channel creation      | `grpc.insecure_channel()`          | `grpc.aio.insecure_channel()`         |
| Handler methods       | `def GetUser(self, request, ctx)`  | `async def GetUser(self, request, ctx)` |
| Streaming yield       | `yield message`                    | `yield message` (in async generator)  |
| Request iteration     | `for req in request_iterator`      | `async for req in request_iterator`   |
| Error handling        | `context.set_code()` + return      | `await context.abort()` (raises)      |
| Concurrency model     | Thread pool (OS threads)           | asyncio event loop (coroutines)       |
| Best for              | CPU-bound, simple services         | I/O-bound, async DB/HTTP calls        |

Use `grpc.aio` when integrating with async frameworks (FastAPI, asyncpg) or when handling many concurrent I/O-bound requests.

---

### Q20. How do you deploy gRPC services?

**Answer**: Common deployment patterns:

1. **Docker**: Package server as a Docker image. Expose port 50051.
   ```dockerfile
   FROM python:3.11-slim
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   EXPOSE 50051
   CMD ["python", "server.py"]
   ```

2. **Kubernetes**: Deploy as a Kubernetes Deployment with a headless Service (for client-side load balancing) or a regular Service with an L7 ingress that supports gRPC (Envoy, nginx with gRPC module).

3. **Health checks**: Implement the gRPC Health Checking Protocol so Kubernetes can do liveness/readiness probes:
   ```yaml
   livenessProbe:
     grpc:
       port: 50051
   ```

4. **Service mesh**: Istio or Linkerd provide transparent load balancing, mTLS, observability, and traffic management for gRPC.

---

### Q21. How does gRPC handle serialization and deserialization?

**Answer**: gRPC uses Protocol Buffers for serialization:

1. **Schema-based**: Messages are defined in `.proto` files with explicit types and field numbers
2. **Binary encoding**: Data is encoded using variable-length integers (varints) for numbers and length-delimited encoding for strings/bytes
3. **Field tags**: Each field is identified by its field number + wire type (not by name), making it very compact
4. **Code generation**: The `protoc` compiler generates language-specific serialization/deserialization code that is optimized at compile time

Example of how `int32 id = 1` with value `42` is encoded:
```
Wire: 08 2A  (2 bytes)
  08 = field 1, wire type 0 (varint)
  2A = 42 in varint encoding

Compared to JSON: {"id": 42}  (10 bytes)
```

This is why Protobuf is ~5-10x smaller than JSON.

---

### Q22. What is the role of the Channel in gRPC?

**Answer**: A Channel is the client-side abstraction that represents a connection to a gRPC server. It handles:

- **Connection management**: Establishes and maintains HTTP/2 connections
- **Connection pooling**: A single channel multiplexes many concurrent RPCs over one connection
- **Name resolution**: Resolves DNS names to IP addresses
- **Load balancing**: Can distribute RPCs across multiple server backends
- **Reconnection**: Automatically reconnects when the connection is lost
- **Health monitoring**: Tracks connection state (IDLE, CONNECTING, READY, TRANSIENT_FAILURE, SHUTDOWN)

Best practice: create **one channel per target server** and reuse it for all stubs. Channels are thread-safe and designed to be long-lived.

```python
# Good: one channel, multiple stubs
channel = grpc.insecure_channel("server:50051")
user_stub = UserServiceStub(channel)
order_stub = OrderServiceStub(channel)
```

---

### Q23. How do you use gRPC for ML model serving?

**Answer**: gRPC is the standard protocol for ML model serving:

1. **TensorFlow Serving**: Uses gRPC natively for prediction requests
2. **Triton Inference Server (NVIDIA)**: gRPC API for inference
3. **Custom model serving**: Define your own service

```protobuf
service MLService {
    // Unary: single prediction
    rpc Predict(PredictRequest) returns (PredictResponse);

    // Server streaming: LLM token-by-token generation
    rpc StreamGenerate(GenerateRequest) returns (stream Token);

    // Client streaming: batch of images for classification
    rpc BatchClassify(stream ImageRequest) returns (BatchResult);
}

message PredictRequest {
    repeated float features = 1;  // input feature vector
    string model_name = 2;
}

message PredictResponse {
    repeated float predictions = 1;
    float confidence = 2;
    int64 latency_ms = 3;
}
```

gRPC is preferred for ML serving because of its low latency, efficient binary serialization of large tensors, and native streaming support (essential for LLM token generation).

---

### Q24. What are the main challenges with gRPC?

**Answer**:

1. **Browser support**: Cannot be called directly from browsers (need gRPC-Web + proxy)
2. **Learning curve**: Protobuf, `.proto` compilation, generated code concepts are unfamiliar to many
3. **Human readability**: Binary format makes debugging harder (cannot just `curl` an endpoint and read JSON)
4. **HTTP caching**: No built-in HTTP caching support (unlike REST with ETag/Cache-Control)
5. **Tooling maturity**: Fewer tools than REST (no Postman-like experience, though Evans and BloomRPC exist)
6. **Load balancing**: HTTP/2 persistent connections make traditional L4 load balancing ineffective; need L7 or client-side balancing
7. **Error handling**: gRPC status codes are less granular than HTTP status codes (only 16 vs. many HTTP codes)
8. **Firewall issues**: Some corporate firewalls block HTTP/2 or non-standard ports

---

### Q25. How do you handle large messages in gRPC?

**Answer**: By default, gRPC limits messages to 4MB. For larger payloads:

1. **Increase the limit** (if messages are occasionally large):
   ```python
   server = grpc.server(
       executor,
       options=[
           ("grpc.max_send_message_length", 100 * 1024 * 1024),     # 100MB
           ("grpc.max_receive_message_length", 100 * 1024 * 1024),
       ],
   )
   ```

2. **Use streaming** (preferred for very large data):
   ```protobuf
   rpc UploadFile(stream FileChunk) returns (UploadResult);

   message FileChunk {
       bytes data = 1;      // e.g., 64KB chunks
       string filename = 2;
       int64 offset = 3;
   }
   ```

3. **Chunking**: Split large data into smaller chunks and stream them. This is more memory-efficient and allows progress tracking.

---

### Q26. What are well-known Protobuf types?

**Answer**: Google provides a library of commonly used message types in `google.protobuf`:

```protobuf
import "google/protobuf/timestamp.proto";    // Timestamp (seconds + nanos)
import "google/protobuf/duration.proto";     // Duration (seconds + nanos)
import "google/protobuf/empty.proto";        // Empty message (for no payload)
import "google/protobuf/wrappers.proto";     // Nullable scalars (Int32Value, StringValue)
import "google/protobuf/any.proto";          // Dynamic typing (like interface{} / object)
import "google/protobuf/struct.proto";       // JSON-like dynamic structure
import "google/protobuf/field_mask.proto";   // Specify which fields to update (partial updates)
```

Example:
```protobuf
message Event {
    string name = 1;
    google.protobuf.Timestamp created_at = 2;  // instead of string or int64
    google.protobuf.Duration ttl = 3;
}

message UpdateUserRequest {
    User user = 1;
    google.protobuf.FieldMask update_mask = 2;  // ["name", "email"]
}
```

---

### Q27. How does gRPC compare to GraphQL?

**Answer**:

| Aspect          | gRPC                      | GraphQL                        |
|-----------------|---------------------------|--------------------------------|
| Primary use     | Service-to-service (backend) | Client-to-server (frontend)  |
| Data format     | Protobuf (binary)         | JSON                           |
| Query flexibility | Fixed methods           | Client chooses fields          |
| Over-fetching   | Possible                  | Solved (query exactly what you need) |
| Streaming       | Native bidirectional      | Subscriptions (via WebSocket)  |
| Type system     | Protobuf schema           | GraphQL schema                 |
| Code gen        | Built-in                  | Optional (codegen tools exist) |
| Performance     | Very fast (binary)        | Slower (JSON, query parsing)   |

gRPC excels at backend microservice communication; GraphQL excels at flexible frontend data fetching. They solve different problems and can coexist.

---

### Q28. Explain gRPC channel states and connection lifecycle.

**Answer**: A gRPC channel goes through these states:

```
IDLE ──► CONNECTING ──► READY ──► (serving requests)
  ▲                       │
  │                       ▼
  │               TRANSIENT_FAILURE ──► CONNECTING (retry)
  │                       │
  │                       ▼
  └───────────────── SHUTDOWN (terminal)
```

- **IDLE**: No RPC activity; connection not established. First RPC triggers CONNECTING.
- **CONNECTING**: Actively trying to establish a connection.
- **READY**: Connection established; RPCs can be sent.
- **TRANSIENT_FAILURE**: Connection lost; will retry with backoff.
- **SHUTDOWN**: Channel explicitly closed; terminal state.

The channel handles reconnection automatically with exponential backoff. Clients do not need to manage connection state manually.

---

### Q29. How do you implement pagination in gRPC?

**Answer**: Use cursor-based pagination (not offset-based) for consistency:

```protobuf
message ListUsersRequest {
    int32 page_size = 1;       // max items to return (e.g., 50)
    string page_token = 2;     // opaque cursor from previous response
    string order_by = 3;       // e.g., "created_at desc"
}

message ListUsersResponse {
    repeated User users = 1;
    string next_page_token = 2;  // empty if no more pages
    int32 total_count = 3;       // optional: total items matching filter
}
```

```python
# Server implementation
def ListUsers(self, request, context):
    page_size = request.page_size or 20
    cursor = decode_cursor(request.page_token) if request.page_token else None

    users = db.query_users(after=cursor, limit=page_size + 1)

    has_more = len(users) > page_size
    if has_more:
        users = users[:page_size]

    return user_pb2.ListUsersResponse(
        users=[to_proto(u) for u in users],
        next_page_token=encode_cursor(users[-1].id) if has_more else "",
    )
```

---

### Q30. What are best practices for designing gRPC APIs?

**Answer**:

1. **Use separate request/response messages** for each RPC (even if they seem similar). This allows independent evolution.

2. **Follow the `google.api` conventions**: `GetX`, `ListX`, `CreateX`, `UpdateX`, `DeleteX` method naming.

3. **Use FieldMask for partial updates** instead of separate "patch" messages.

4. **Version via packages** (`user.v1`, `user.v2`) not method name changes.

5. **Keep messages small**: Use streaming for large payloads instead of giant messages.

6. **Reserve deleted field numbers**: Never reuse them.

7. **Use well-known types**: `Timestamp`, `Duration`, `Empty`, `FieldMask` instead of reinventing them.

8. **Enum zero values should be UNSPECIFIED**: `MY_ENUM_UNSPECIFIED = 0` as a sentinel.

9. **Return the created/updated resource** from mutating RPCs (not just an ID).

10. **Set deadlines on every client call**: Never make an RPC without a timeout.

11. **Use interceptors for cross-cutting concerns**: Auth, logging, metrics, tracing.

12. **Enable reflection in dev/staging**: Makes debugging much easier.

---

## Summary Cheat Sheet

```
REST                               gRPC
────                               ────
JSON (text)                        Protobuf (binary)
HTTP/1.1 or 2                      HTTP/2 only
Resources + verbs                  Services + methods
OpenAPI (optional)                 .proto (required)
curl / Postman                     grpcurl / Evans
Status codes: 200, 404, 500...     Status codes: OK, NOT_FOUND, INTERNAL...
Middleware                         Interceptors
Public APIs, browsers              Internal APIs, microservices, ML serving

Proto Quick Reference:
  syntax = "proto3";
  package name.v1;
  message Foo { string bar = 1; repeated int32 ids = 2; }
  enum Status { UNSPECIFIED = 0; ACTIVE = 1; }
  service FooService { rpc Get(GetReq) returns (GetRes); }

Four Patterns:
  Unary:           req -> res
  Server stream:   req -> stream res
  Client stream:   stream req -> res
  Bidi stream:     stream req <-> stream res

Python:
  pip install grpcio grpcio-tools
  python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. foo.proto
  Sync:  grpc.server() / grpc.insecure_channel()
  Async: grpc.aio.server() / grpc.aio.insecure_channel()
```
