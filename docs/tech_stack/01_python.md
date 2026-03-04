# 01. Python: Async, Concurrency, Type System & Project Structure

> Comprehensive interview preparation for a Backend AI Engineer role.
> Covers theory, code examples, and 25+ Q&A items.

---

## Table of Contents

1. [GIL (Global Interpreter Lock)](#1-gil-global-interpreter-lock)
2. [Threading vs Multiprocessing vs Asyncio](#2-threading-vs-multiprocessing-vs-asyncio)
3. [Asyncio Deep Dive](#3-asyncio-deep-dive)
4. [aiohttp](#4-aiohttp)
5. [FastAPI](#5-fastapi)
6. [Type Hints](#6-type-hints)
7. [Pydantic](#7-pydantic)
8. [Dependency Injection Patterns](#8-dependency-injection-patterns)
9. [Project Structure for Production](#9-project-structure-for-production)
10. [Q&A Section (25 Questions)](#10-qa-section)

---

## 1. GIL (Global Interpreter Lock)

### What Is the GIL?

The **Global Interpreter Lock** is a mutex in **CPython** that allows only one thread to execute Python bytecode at any given moment, even on multi-core machines. It exists because CPython's memory management (reference counting) is not thread-safe; without the GIL, simultaneous reference-count updates from multiple threads would corrupt object lifetimes.

**Key points:**

- The GIL is a **CPython** implementation detail, not a language specification requirement.
- It protects the interpreter's internal data structures from concurrent access.
- It means CPU-bound Python threads cannot run in true parallel on multiple cores.

### Visual: GIL Behavior

```
CPU-bound work (GIL limits parallelism):
  Thread 1: ████____████____████    (runs, yields GIL, runs, yields...)
  Thread 2: ____████____████____    (takes turns with Thread 1)
  Time:     0   1   2   3   4   5
  Result:   ~Same wall-clock time as single-threaded — no speedup.

I/O-bound work (GIL released during I/O waits):
  Thread 1: ██==wait==██==wait==██  (releases GIL while waiting for I/O)
  Thread 2: __██==wait==██==wait==  (runs during Thread 1's I/O wait)
  Result:   Real concurrency for I/O operations — significant speedup.
```

### How the GIL Works Internally

1. **Tick-based switching (Python 2):** Every 100 bytecode instructions, the running thread released the GIL.
2. **Time-based switching (Python 3.2+):** A thread holds the GIL for a default of **5 ms** (`sys.getswitchinterval()`), then is asked to release it so other threads can run.
3. **I/O release:** Whenever a thread enters a C extension that performs I/O (socket read, file write, `time.sleep`), it **voluntarily releases the GIL**.

```python
import sys

# Default: 0.005 seconds (5 ms)
print(sys.getswitchinterval())

# You can change it (rarely needed):
sys.setswitchinterval(0.001)
```

### CPython vs Other Implementations

| Implementation | Has GIL? | Notes |
|---|---|---|
| **CPython** | Yes | Reference implementation; GIL present. |
| **Jython** | No | Runs on JVM; uses JVM's threading model. |
| **IronPython** | No | Runs on .NET CLR; true multi-threading. |
| **PyPy** | Yes | Has GIL, but JIT compilation makes single-threaded code much faster. PyPy STM was experimental. |
| **GraalPy** | No | Python on GraalVM; no GIL. |

### Python 3.13 Free-Threaded Mode (No-GIL Experimental)

PEP 703 introduced an **experimental** build of CPython 3.13 that disables the GIL entirely.

```bash
# Install the free-threaded build (example with pyenv):
# The free-threaded build has a 't' suffix
pyenv install 3.13.0t

# Or compile from source:
./configure --disable-gil
make
```

**How it works:**

- Reference counting is replaced with **biased reference counting** and **deferred reference counting** for certain objects.
- Per-object **fine-grained locks** protect internal structures instead of one global lock.
- A new `mimalloc`-based allocator provides thread-safe memory allocation.

**Trade-offs:**

- Single-threaded performance may be ~5-10% slower due to atomic operations overhead.
- Many C extensions need to be updated (they assumed the GIL protected shared state).
- It is **opt-in** and **experimental** in 3.13; expected to stabilize over 3.14-3.15.

```python
import sys

# Check if running free-threaded build:
print(sys._is_gil_enabled())  # False in no-GIL build
```

### GIL Demonstration: CPU-bound vs I/O-bound

```python
import threading
import time
import math

# --- CPU-bound task ---
def cpu_work(n: int) -> float:
    """Compute sum of square roots (pure CPU)."""
    total = 0.0
    for i in range(n):
        total += math.sqrt(i)
    return total

N = 5_000_000

# Single-threaded
start = time.perf_counter()
cpu_work(N)
cpu_work(N)
print(f"Single-threaded CPU: {time.perf_counter() - start:.2f}s")

# Multi-threaded (GIL prevents true parallelism)
start = time.perf_counter()
t1 = threading.Thread(target=cpu_work, args=(N,))
t2 = threading.Thread(target=cpu_work, args=(N,))
t1.start(); t2.start()
t1.join(); t2.join()
print(f"Multi-threaded CPU:  {time.perf_counter() - start:.2f}s")
# Result: roughly the SAME time (or even slower due to context switching).

# --- I/O-bound task ---
def io_work(seconds: float) -> None:
    """Simulate I/O wait."""
    time.sleep(seconds)

# Single-threaded
start = time.perf_counter()
io_work(1.0)
io_work(1.0)
print(f"Single-threaded I/O: {time.perf_counter() - start:.2f}s")  # ~2.0s

# Multi-threaded (GIL released during sleep)
start = time.perf_counter()
t1 = threading.Thread(target=io_work, args=(1.0,))
t2 = threading.Thread(target=io_work, args=(1.0,))
t1.start(); t2.start()
t1.join(); t2.join()
print(f"Multi-threaded I/O:  {time.perf_counter() - start:.2f}s")  # ~1.0s
```

---

## 2. Threading vs Multiprocessing vs Asyncio

### Visual Comparison

```
Threading (concurrent, shared memory, one process):
  Process ┌─ Thread 1: ██░░██░░██
          └─ Thread 2: ░░██░░██░░
  - Threads share the same memory space.
  - GIL limits CPU parallelism in CPython.
  - Good for I/O-bound tasks.

Multiprocessing (parallel, separate memory):
  Process 1: ██████████  ← own memory, own GIL
  Process 2: ██████████  ← own memory, own GIL
  - Each process has its own Python interpreter and GIL.
  - True parallelism on multi-core CPUs.
  - Good for CPU-bound tasks.

Asyncio (cooperative, single thread, event loop):
  Event Loop: ──Task1──await──Task2──await──Task3──await──
  - Single thread, no OS context switching.
  - Tasks voluntarily yield at `await` points.
  - Excellent for high-concurrency I/O (thousands of connections).
```

### When to Use Each

| Scenario | Recommended | Why |
|---|---|---|
| **CPU-heavy computation** (ML training, image processing) | `multiprocessing` | Bypasses GIL; uses all cores. |
| **I/O-heavy, moderate concurrency** (file ops, subprocess calls) | `threading` | Simple API; GIL released during I/O. |
| **I/O-heavy, high concurrency** (web servers, API calls, scrapers) | `asyncio` | Single-thread handles thousands of tasks; low overhead. |
| **Mixed I/O + CPU** | `asyncio` + `ProcessPoolExecutor` | Async for I/O; offload CPU to separate processes. |

### Threading Example

```python
import threading
import time
from typing import List

results: List[str] = []
lock = threading.Lock()

def fetch_url(url: str) -> None:
    """Simulate fetching a URL."""
    time.sleep(1)  # Simulate network I/O
    with lock:
        results.append(f"Fetched {url}")

urls = [f"https://example.com/page/{i}" for i in range(5)]

threads = [threading.Thread(target=fetch_url, args=(url,)) for url in urls]
start = time.perf_counter()
for t in threads:
    t.start()
for t in threads:
    t.join()

elapsed = time.perf_counter() - start
print(f"Threading: {len(results)} results in {elapsed:.2f}s")  # ~1s, not 5s
```

### Multiprocessing Example

```python
import multiprocessing
import math
import time

def heavy_computation(n: int) -> float:
    """CPU-bound work."""
    return sum(math.sqrt(i) for i in range(n))

if __name__ == "__main__":
    N = 10_000_000

    # Single process
    start = time.perf_counter()
    heavy_computation(N)
    heavy_computation(N)
    heavy_computation(N)
    heavy_computation(N)
    print(f"Single process: {time.perf_counter() - start:.2f}s")

    # Multiprocessing (4 workers)
    start = time.perf_counter()
    with multiprocessing.Pool(4) as pool:
        results = pool.map(heavy_computation, [N, N, N, N])
    print(f"4 processes:    {time.perf_counter() - start:.2f}s")
    # Expect roughly 4x speedup on a 4+ core machine.
```

### Asyncio Example

```python
import asyncio
import time

async def fetch_data(item_id: int) -> dict:
    """Simulate async I/O operation."""
    await asyncio.sleep(1)  # Non-blocking sleep
    return {"id": item_id, "data": f"result_{item_id}"}

async def main() -> None:
    start = time.perf_counter()

    # Launch 10 tasks concurrently
    tasks = [fetch_data(i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start
    print(f"Asyncio: {len(results)} results in {elapsed:.2f}s")  # ~1s, not 10s

asyncio.run(main())
```

### concurrent.futures: Unified High-Level API

`concurrent.futures` provides a uniform interface for both threading and multiprocessing via `Executor` objects.

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import math

def io_task(url: str) -> str:
    time.sleep(1)
    return f"Fetched {url}"

def cpu_task(n: int) -> float:
    return sum(math.sqrt(i) for i in range(n))

# --- ThreadPoolExecutor for I/O-bound ---
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(io_task, f"url_{i}"): i for i in range(5)}
    for future in as_completed(futures):
        idx = futures[future]
        print(f"Task {idx}: {future.result()}")

# --- ProcessPoolExecutor for CPU-bound ---
if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(cpu_task, 5_000_000): i for i in range(4)}
        for future in as_completed(futures):
            idx = futures[future]
            print(f"Task {idx}: result = {future.result():.2f}")
```

**Key differences between the two executors:**

| Feature | `ThreadPoolExecutor` | `ProcessPoolExecutor` |
|---|---|---|
| Memory | Shared | Separate (data must be pickled) |
| GIL | Affected | Not affected (each process has its own) |
| Startup cost | Low | High (process creation + pickling) |
| Best for | I/O-bound | CPU-bound |
| Debugging | Easier (shared state visible) | Harder (separate address spaces) |

### Integrating concurrent.futures with Asyncio

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor
import math

def cpu_heavy(n: int) -> float:
    """Runs in a separate process to avoid blocking the event loop."""
    return sum(math.sqrt(i) for i in range(n))

async def main() -> None:
    loop = asyncio.get_running_loop()

    # Offload CPU work to a process pool
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = await asyncio.gather(
            loop.run_in_executor(pool, cpu_heavy, 5_000_000),
            loop.run_in_executor(pool, cpu_heavy, 5_000_000),
            loop.run_in_executor(pool, cpu_heavy, 5_000_000),
            loop.run_in_executor(pool, cpu_heavy, 5_000_000),
        )
    print(f"Results: {[f'{r:.2f}' for r in results]}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 3. Asyncio Deep Dive

### Event Loop Mechanics

The **event loop** is the core of asyncio. It runs in a single thread and manages a queue of **coroutines** (tasks). At each iteration it:

1. Checks for completed I/O operations (via `select`, `epoll`, or `kqueue`).
2. Runs callbacks registered for those completions.
3. Advances coroutines that were awaiting those results.
4. Schedules newly created tasks.

```
┌─────────────────────────────────────────┐
│             Event Loop Cycle            │
│                                         │
│  1. Poll for I/O events (epoll/kqueue)  │
│  2. Run ready callbacks                 │
│  3. Advance coroutines past `await`     │
│  4. Schedule new tasks                  │
│  5. Handle timers / scheduled calls     │
│  6. Repeat                              │
└─────────────────────────────────────────┘
```

### async/await Syntax

```python
import asyncio

# A coroutine function (defined with 'async def'):
async def greet(name: str) -> str:
    await asyncio.sleep(0.1)  # Yield control to the event loop
    return f"Hello, {name}!"

# Calling a coroutine function returns a coroutine OBJECT (not the result):
coro = greet("Alice")  # <coroutine object greet at 0x...>

# To actually run it, you must await it or schedule it:
async def main() -> None:
    result = await greet("Alice")
    print(result)

asyncio.run(main())
```

### Tasks: create_task, gather, wait

**`asyncio.create_task()`** schedules a coroutine to run concurrently on the event loop. It returns a `Task` object that wraps the coroutine.

```python
import asyncio

async def fetch(url: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"Data from {url}"

async def main() -> None:
    # --- create_task: schedule and run concurrently ---
    task1 = asyncio.create_task(fetch("api/users", 1.0))
    task2 = asyncio.create_task(fetch("api/orders", 0.5))

    # Both tasks are running NOW. We can do other work here.
    print("Tasks started...")

    # Await results
    result1 = await task1
    result2 = await task2
    print(result1, result2)

asyncio.run(main())
```

**`asyncio.gather()`** runs multiple awaitables concurrently and returns results in order.

```python
async def main() -> None:
    results = await asyncio.gather(
        fetch("api/users", 1.0),
        fetch("api/orders", 0.5),
        fetch("api/products", 0.8),
    )
    # results is a list in the SAME order as the arguments:
    # ["Data from api/users", "Data from api/orders", "Data from api/products"]
    print(results)

    # With return_exceptions=True, exceptions are returned as values
    # instead of being raised:
    results = await asyncio.gather(
        fetch("api/users", 1.0),
        failing_coroutine(),
        return_exceptions=True,
    )
    # results[1] will be an Exception object
```

**`asyncio.wait()`** provides fine-grained control over completion.

```python
async def main() -> None:
    tasks = {
        asyncio.create_task(fetch("api/users", 2.0), name="users"),
        asyncio.create_task(fetch("api/orders", 0.5), name="orders"),
        asyncio.create_task(fetch("api/products", 1.0), name="products"),
    }

    # Wait for the FIRST task to complete:
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in done:
        print(f"Completed first: {task.get_name()} -> {task.result()}")

    # Wait for ALL remaining tasks:
    done2, _ = await asyncio.wait(pending, return_when=asyncio.ALL_COMPLETED)
    for task in done2:
        print(f"Completed: {task.get_name()} -> {task.result()}")
```

### TaskGroup (Python 3.11+)

`TaskGroup` provides structured concurrency with proper exception handling.

```python
async def main() -> None:
    results = []

    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(fetch("api/users", 1.0))
        task2 = tg.create_task(fetch("api/orders", 0.5))
        task3 = tg.create_task(fetch("api/products", 0.8))

    # All tasks are guaranteed to be done here
    results = [task1.result(), task2.result(), task3.result()]
    print(results)

    # If ANY task raises, TaskGroup cancels all other tasks and raises
    # an ExceptionGroup containing all exceptions.
```

### Synchronization Primitives

#### asyncio.Lock

```python
import asyncio

shared_resource: list[str] = []
lock = asyncio.Lock()

async def safe_append(value: str) -> None:
    async with lock:
        # Only one coroutine at a time can execute this block
        shared_resource.append(value)
        await asyncio.sleep(0.1)  # Simulate some async work
        print(f"Appended {value}, total: {len(shared_resource)}")

async def main() -> None:
    await asyncio.gather(*(safe_append(f"item_{i}") for i in range(5)))
    print(f"Final: {shared_resource}")

asyncio.run(main())
```

#### asyncio.Semaphore

Limits the number of concurrent accesses to a resource (e.g., rate-limiting API calls).

```python
import asyncio

semaphore = asyncio.Semaphore(3)  # Max 3 concurrent

async def rate_limited_fetch(url: str) -> str:
    async with semaphore:
        print(f"Fetching {url} (active: {3 - semaphore._value}/3)")
        await asyncio.sleep(1)  # Simulate network I/O
        return f"Result from {url}"

async def main() -> None:
    urls = [f"https://api.example.com/item/{i}" for i in range(10)]
    results = await asyncio.gather(*(rate_limited_fetch(url) for url in urls))
    print(f"Got {len(results)} results")

asyncio.run(main())
```

#### asyncio.Queue

Producer-consumer pattern for async pipelines.

```python
import asyncio
import random

async def producer(queue: asyncio.Queue[int], name: str) -> None:
    for i in range(5):
        item = random.randint(1, 100)
        await queue.put(item)
        print(f"[{name}] Produced: {item}")
        await asyncio.sleep(random.uniform(0.1, 0.5))
    await queue.put(None)  # Sentinel to signal completion

async def consumer(queue: asyncio.Queue[int | None], name: str) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        print(f"  [{name}] Consumed: {item}")
        await asyncio.sleep(random.uniform(0.1, 0.3))
        queue.task_done()

async def main() -> None:
    queue: asyncio.Queue[int | None] = asyncio.Queue(maxsize=10)

    producers = [asyncio.create_task(producer(queue, f"P{i}")) for i in range(2)]
    consumers = [asyncio.create_task(consumer(queue, f"C{i}")) for i in range(3)]

    await asyncio.gather(*producers)
    # Signal consumers to stop (one sentinel per consumer)
    for _ in range(3):
        await queue.put(None)
    await asyncio.gather(*consumers)

asyncio.run(main())
```

### Common Pitfalls

#### 1. Blocking the Event Loop

```python
import asyncio
import time

# BAD: blocks the entire event loop
async def bad_handler() -> None:
    time.sleep(5)  # This is synchronous! No other task can run.

# GOOD: use asyncio.sleep or run in executor
async def good_handler() -> None:
    await asyncio.sleep(5)  # Non-blocking

# GOOD: offload blocking code to a thread
async def good_handler_v2() -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, time.sleep, 5)  # Runs in thread pool
```

#### 2. Forgetting `await`

```python
async def fetch_data() -> dict:
    await asyncio.sleep(1)
    return {"key": "value"}

async def main() -> None:
    # BAD: missing await returns a coroutine object, not the result
    result = fetch_data()  # RuntimeWarning: coroutine was never awaited
    print(result)  # <coroutine object fetch_data at 0x...>

    # GOOD:
    result = await fetch_data()
    print(result)  # {"key": "value"}
```

#### 3. Fire-and-Forget Task Gets Garbage Collected

```python
async def background_work() -> None:
    await asyncio.sleep(10)
    print("Done!")  # May never print!

async def main() -> None:
    # BAD: task reference is lost, may be garbage collected
    asyncio.create_task(background_work())

    # GOOD: keep a reference
    task = asyncio.create_task(background_work())
    # ... do other work ...
    await task  # Ensure it completes

    # GOOD (alternative): keep tasks in a set
    background_tasks: set[asyncio.Task] = set()
    task = asyncio.create_task(background_work())
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
```

#### 4. Using `asyncio.gather` without Error Handling

```python
async def may_fail(i: int) -> str:
    if i == 2:
        raise ValueError("Oops")
    return f"ok_{i}"

async def main() -> None:
    # If any coroutine raises, gather raises immediately
    # and the other results are lost.
    try:
        results = await asyncio.gather(
            may_fail(1), may_fail(2), may_fail(3)
        )
    except ValueError as e:
        print(f"Error: {e}")  # We lose results from may_fail(1) and may_fail(3)

    # BETTER: use return_exceptions=True
    results = await asyncio.gather(
        may_fail(1), may_fail(2), may_fail(3),
        return_exceptions=True,
    )
    for r in results:
        if isinstance(r, Exception):
            print(f"Error: {r}")
        else:
            print(f"Result: {r}")
```

### Code Example: Async Web Scraper

```python
import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Optional

@dataclass
class PageResult:
    url: str
    status: int
    content_length: int
    title: Optional[str] = None

async def fetch_page(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
) -> PageResult:
    """Fetch a single page with rate limiting."""
    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                html = await resp.text()
                # Simple title extraction
                title = None
                if "<title>" in html:
                    start = html.index("<title>") + 7
                    end = html.index("</title>", start)
                    title = html[start:end].strip()
                return PageResult(
                    url=url,
                    status=resp.status,
                    content_length=len(html),
                    title=title,
                )
        except Exception as e:
            return PageResult(url=url, status=0, content_length=0, title=f"Error: {e}")

async def scrape(urls: list[str], max_concurrent: int = 5) -> list[PageResult]:
    """Scrape multiple URLs concurrently with rate limiting."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_page(session, url, semaphore) for url in urls]
        results = await asyncio.gather(*tasks)

    return list(results)

async def main() -> None:
    urls = [
        "https://httpbin.org/html",
        "https://httpbin.org/json",
        "https://httpbin.org/delay/1",
        "https://httpbin.org/status/404",
        "https://httpbin.org/get",
    ]

    print(f"Scraping {len(urls)} URLs...")
    results = await scrape(urls, max_concurrent=3)

    for r in results:
        print(f"  {r.status} | {r.content_length:>6} bytes | {r.url} | {r.title}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4. aiohttp

### Overview

`aiohttp` is an async HTTP client/server framework built on top of asyncio. It provides both a **client** for making HTTP requests and a **server** for building web applications.

### Client: Session Management

```python
import aiohttp
import asyncio

async def client_example() -> None:
    # GOOD: reuse a single session for multiple requests (connection pooling)
    async with aiohttp.ClientSession() as session:

        # GET request
        async with session.get("https://httpbin.org/get") as resp:
            print(f"Status: {resp.status}")
            data = await resp.json()
            print(data)

        # POST request with JSON body
        async with session.post(
            "https://httpbin.org/post",
            json={"key": "value"},
        ) as resp:
            data = await resp.json()
            print(data)

        # POST with form data
        async with session.post(
            "https://httpbin.org/post",
            data={"field1": "value1", "field2": "value2"},
        ) as resp:
            data = await resp.json()
            print(data)

        # Custom headers and timeout
        async with session.get(
            "https://httpbin.org/headers",
            headers={"Authorization": "Bearer token123"},
            timeout=aiohttp.ClientTimeout(total=30, connect=5),
        ) as resp:
            data = await resp.json()
            print(data)

asyncio.run(client_example())
```

**Session best practices:**

- Create **one session** per application or per logical group of requests.
- Sessions maintain a **connection pool** and **cookie jar** internally.
- Always use `async with` to ensure proper cleanup.
- Do **not** create a new session per request (expensive).

```python
# BAD: new session per request
async def bad_fetch(url: str) -> dict:
    async with aiohttp.ClientSession() as session:  # Wasteful!
        async with session.get(url) as resp:
            return await resp.json()

# GOOD: pass shared session
async def good_fetch(session: aiohttp.ClientSession, url: str) -> dict:
    async with session.get(url) as resp:
        return await resp.json()
```

### Server Setup

```python
from aiohttp import web
import asyncio

# Route handlers
async def handle_root(request: web.Request) -> web.Response:
    return web.json_response({"message": "Hello, World!"})

async def handle_user(request: web.Request) -> web.Response:
    user_id = request.match_info["user_id"]
    return web.json_response({"user_id": user_id})

async def handle_create_user(request: web.Request) -> web.Response:
    body = await request.json()
    return web.json_response({"created": body}, status=201)

# Application setup
app = web.Application()
app.router.add_get("/", handle_root)
app.router.add_get("/users/{user_id}", handle_user)
app.router.add_post("/users", handle_create_user)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=8080)
```

### Middleware

```python
from aiohttp import web
import time
import logging

logger = logging.getLogger(__name__)

# Middleware is a decorator/function that wraps every request handler.

@web.middleware
async def logging_middleware(
    request: web.Request,
    handler: web.RequestHandler,
) -> web.StreamResponse:
    """Log request method, path, and response time."""
    start = time.perf_counter()
    try:
        response = await handler(request)
        elapsed = time.perf_counter() - start
        logger.info(
            "%s %s -> %s (%.3fs)",
            request.method, request.path, response.status, elapsed,
        )
        return response
    except web.HTTPException as e:
        elapsed = time.perf_counter() - start
        logger.warning(
            "%s %s -> %s (%.3fs)",
            request.method, request.path, e.status, elapsed,
        )
        raise

@web.middleware
async def error_middleware(
    request: web.Request,
    handler: web.RequestHandler,
) -> web.StreamResponse:
    """Catch unhandled exceptions and return JSON error responses."""
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error")
        return web.json_response(
            {"error": str(e)},
            status=500,
        )

# Apply middleware (executed in order: logging -> error -> handler)
app = web.Application(middlewares=[logging_middleware, error_middleware])
```

### Comparison: aiohttp vs requests

| Feature | `requests` | `aiohttp` |
|---|---|---|
| **Paradigm** | Synchronous | Asynchronous (asyncio) |
| **Concurrency** | One request at a time (unless threaded) | Thousands of concurrent requests |
| **Server** | No (client only) | Yes (client + server) |
| **WebSockets** | No (needs `websocket-client`) | Yes (built-in) |
| **Session** | `requests.Session()` | `aiohttp.ClientSession()` |
| **Streaming** | `iter_content()` | `resp.content.read(n)` |
| **API simplicity** | Simpler, synchronous | More verbose, requires `async/await` |
| **Best for** | Scripts, simple tasks | High-concurrency services |

```python
# requests (synchronous)
import requests

def sync_fetch() -> None:
    resp = requests.get("https://httpbin.org/get")
    print(resp.json())

# aiohttp (asynchronous)
import aiohttp
import asyncio

async def async_fetch() -> None:
    async with aiohttp.ClientSession() as session:
        async with session.get("https://httpbin.org/get") as resp:
            print(await resp.json())

asyncio.run(async_fetch())
```

---

## 5. FastAPI

### Overview

FastAPI is a modern, high-performance web framework built on Starlette (ASGI) and Pydantic. It provides automatic OpenAPI docs, data validation, and native async support.

### App Structure and Routing

```python
from fastapi import FastAPI, Query, Path, Body, HTTPException, status
from pydantic import BaseModel

app = FastAPI(title="My API", version="1.0.0")

# --- Path parameters ---
@app.get("/users/{user_id}")
async def get_user(
    user_id: int = Path(..., ge=1, description="The ID of the user"),
) -> dict:
    return {"user_id": user_id}

# --- Query parameters ---
@app.get("/users")
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    search: str | None = Query(None, min_length=1),
) -> dict:
    return {"skip": skip, "limit": limit, "search": search}

# --- Request body ---
class UserCreate(BaseModel):
    name: str
    email: str
    age: int | None = None

class UserResponse(BaseModel):
    id: int
    name: str
    email: str

@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate) -> UserResponse:
    # In real code, save to DB
    return UserResponse(id=1, name=user.name, email=user.email)

# --- Error handling ---
@app.get("/items/{item_id}")
async def get_item(item_id: int) -> dict:
    if item_id == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item not found",
        )
    return {"item_id": item_id}
```

### Dependency Injection with `Depends`

FastAPI's DI system is one of its most powerful features. Dependencies are functions (sync or async) that are called before the route handler and whose return values are injected as parameters.

```python
from fastapi import FastAPI, Depends, HTTPException, Header, status
from typing import Annotated

app = FastAPI()

# --- Simple dependency ---
async def get_db():
    """Dependency that provides a database session."""
    db = AsyncSessionLocal()
    try:
        yield db  # 'yield' makes it a context-manager dependency
    finally:
        await db.close()

# --- Auth dependency ---
async def get_current_user(
    authorization: str = Header(...),
) -> dict:
    """Extract and validate the current user from the auth header."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header",
        )
    token = authorization.removeprefix("Bearer ")
    # Validate token, fetch user...
    user = {"id": 1, "name": "Alice", "token": token}
    return user

# --- Dependency that depends on another dependency ---
async def get_admin_user(
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Ensure the current user is an admin."""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user

# --- Using dependencies in routes ---
@app.get("/profile")
async def get_profile(
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    return {"user": user}

@app.get("/admin/dashboard")
async def admin_dashboard(
    admin: Annotated[dict, Depends(get_admin_user)],
) -> dict:
    return {"admin": admin, "dashboard": "secret data"}
```

### Background Tasks

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI()

def send_email(to: str, subject: str, body: str) -> None:
    """Blocking function that runs in the background after the response is sent."""
    import time
    time.sleep(2)  # Simulate sending email
    print(f"Email sent to {to}: {subject}")

class UserSignup(BaseModel):
    name: str
    email: str

@app.post("/signup")
async def signup(
    user: UserSignup,
    background_tasks: BackgroundTasks,
) -> dict:
    # Enqueue email sending to run AFTER the response
    background_tasks.add_task(
        send_email,
        to=user.email,
        subject="Welcome!",
        body=f"Hi {user.name}, welcome to our platform.",
    )
    # Response is returned immediately
    return {"message": f"User {user.name} signed up. Welcome email queued."}
```

### Middleware

```python
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import time
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

# --- Built-in CORS middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://frontend.example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Custom middleware ---
@app.middleware("http")
async def request_timing_middleware(request: Request, call_next) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    response.headers["X-Process-Time"] = f"{elapsed:.4f}"
    logger.info(
        "%s %s -> %s (%.4fs)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response

# --- Exception handler ---
from fastapi.responses import JSONResponse

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )
```

### WebSocket Support

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List

app = FastAPI()

class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str) -> None:
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str) -> None:
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"Client {client_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client {client_id} left the chat")
```

### Lifespan Events

Lifespan events replace the deprecated `@app.on_event("startup")` and `@app.on_event("shutdown")` decorators. They use an async context manager.

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from typing import AsyncIterator

# Shared resources
ml_model = None
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown."""
    # --- Startup ---
    global ml_model, db_pool
    print("Loading ML model...")
    ml_model = load_model("model.pkl")  # Heavy initialization
    db_pool = await create_pool("postgresql://...")
    print("Application ready.")

    yield  # Application runs here

    # --- Shutdown ---
    print("Shutting down...")
    await db_pool.close()
    ml_model = None
    print("Cleanup complete.")

app = FastAPI(lifespan=lifespan)

@app.get("/predict")
async def predict(text: str) -> dict:
    result = ml_model.predict(text)
    return {"prediction": result}
```

### Production-Ready API Structure Example

```python
"""
Complete example showing how everything fits together.
File: app/main.py
"""

from fastapi import FastAPI
from contextlib import asynccontextmanager
from typing import AsyncIterator

from app.api.v1.router import api_router
from app.core.config import settings
from app.db.session import engine

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup: create tables, warm caches, etc.
    yield
    # Shutdown: close connections, flush buffers, etc.
    await engine.dispose()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
)

app.include_router(api_router, prefix="/api/v1")
```

---

## 6. Type Hints

### Basic Types

```python
# Built-in types (Python 3.10+ syntax using |)
name: str = "Alice"
age: int = 30
score: float = 95.5
active: bool = True
data: bytes = b"hello"

# Collections (Python 3.9+ supports built-in generics)
names: list[str] = ["Alice", "Bob"]
scores: dict[str, float] = {"Alice": 95.5}
unique_ids: set[int] = {1, 2, 3}
coordinates: tuple[float, float] = (1.0, 2.0)
variable_tuple: tuple[int, ...] = (1, 2, 3, 4)  # Variable length

# Optional / Union (Python 3.10+)
maybe_name: str | None = None      # Equivalent to Optional[str]
id_or_name: int | str = "user_1"   # Equivalent to Union[int, str]

# Pre-3.10 equivalents (still commonly seen):
from typing import Optional, Union, List, Dict, Tuple, Set

maybe_name_old: Optional[str] = None
id_or_name_old: Union[int, str] = "user_1"
names_old: List[str] = ["Alice"]
```

### Advanced Types

```python
from typing import (
    Literal, TypeVar, Generic, Callable, Awaitable,
    TypeAlias, TypeGuard, Never, Self,
)

# --- Literal: restrict values to specific literals ---
Mode = Literal["train", "eval", "predict"]

def run_model(mode: Mode) -> None:
    ...  # mypy ensures only "train", "eval", or "predict" is passed

# --- TypeVar: generic type variables ---
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

def first(items: list[T]) -> T:
    """Return the first item. Return type matches the element type."""
    return items[0]

# Bounded TypeVar: restrict to subclasses
from numbers import Number
N = TypeVar("N", bound=Number)

def double(n: N) -> N:
    return n * 2  # type: ignore

# --- Generic classes ---
class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

    def peek(self) -> T:
        return self._items[-1]

int_stack: Stack[int] = Stack()
int_stack.push(42)
value: int = int_stack.pop()

# --- Callable: function signatures ---
Callback = Callable[[str, int], bool]  # (str, int) -> bool
AsyncCallback = Callable[[str], Awaitable[dict]]  # async (str) -> dict

def apply(func: Callback, name: str, age: int) -> bool:
    return func(name, age)

# --- TypeAlias (Python 3.10+) ---
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

# --- TypeGuard (Python 3.10+): narrowing ---
def is_string_list(val: list[object]) -> TypeGuard[list[str]]:
    return all(isinstance(x, str) for x in val)

def process(items: list[object]) -> None:
    if is_string_list(items):
        # Here, mypy knows items is list[str]
        print(", ".join(items))
```

### Protocol Classes (Structural Subtyping)

Protocols define interfaces via structure (duck typing), not inheritance. A class satisfies a Protocol if it has the required methods/attributes, without needing to explicitly inherit from it.

```python
from typing import Protocol, runtime_checkable

# --- Define a Protocol ---
@runtime_checkable  # Allows isinstance() checks (optional)
class Drawable(Protocol):
    def draw(self, x: int, y: int) -> None:
        ...

class Circle:
    """No explicit inheritance from Drawable, but satisfies it."""
    def draw(self, x: int, y: int) -> None:
        print(f"Drawing circle at ({x}, {y})")

class Square:
    def draw(self, x: int, y: int) -> None:
        print(f"Drawing square at ({x}, {y})")

class NotDrawable:
    def render(self, x: int, y: int) -> None:
        print("I don't have draw()")

def render_shape(shape: Drawable) -> None:
    """Accepts anything with a draw(x, y) method."""
    shape.draw(10, 20)

render_shape(Circle())  # OK
render_shape(Square())  # OK
# render_shape(NotDrawable())  # mypy error: NotDrawable lacks draw()

# runtime_checkable allows isinstance checks:
print(isinstance(Circle(), Drawable))      # True
print(isinstance(NotDrawable(), Drawable)) # False

# --- Protocol with properties and multiple methods ---
class Repository(Protocol[T]):
    async def get(self, id: int) -> T | None:
        ...

    async def list(self, skip: int = 0, limit: int = 10) -> list[T]:
        ...

    async def create(self, entity: T) -> T:
        ...

    async def delete(self, id: int) -> bool:
        ...
```

### TypedDict

Typed dictionaries allow you to declare the expected structure of a dictionary with specific key-value types.

```python
from typing import TypedDict, NotRequired, Required

class UserDict(TypedDict):
    id: int
    name: str
    email: str
    age: NotRequired[int]  # Optional key (Python 3.11+)

# All keys are required unless NotRequired is used
user: UserDict = {"id": 1, "name": "Alice", "email": "alice@example.com"}
user_with_age: UserDict = {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 30}

# TypedDict with total=False (all keys optional by default)
class PartialUpdate(TypedDict, total=False):
    name: str
    email: str
    age: int
    active: Required[bool]  # This one IS required

# Inheritance
class AdminDict(UserDict):
    role: str
    permissions: list[str]
```

### Runtime vs Static Checking

```python
# Type hints are NOT enforced at runtime by Python itself.
# They are used by STATIC checkers like mypy, pyright, etc.

def greet(name: str) -> str:
    return f"Hello, {name}"

# This works at runtime (no error!) but mypy catches it:
greet(42)  # mypy error: Argument 1 has incompatible type "int"; expected "str"

# --- Running mypy ---
# pip install mypy
# mypy my_module.py --strict

# Common mypy flags:
# --strict                Enable all optional checks
# --disallow-untyped-defs Require type hints on all functions
# --no-implicit-optional  Don't treat 'x: str = None' as Optional[str]
# --warn-return-any       Warn when returning Any
# --check-untyped-defs    Check bodies of untyped functions too
```

---

## 7. Pydantic

### BaseModel Basics

```python
from pydantic import BaseModel, Field
from datetime import datetime

class User(BaseModel):
    id: int
    name: str = Field(..., min_length=1, max_length=100, description="User's full name")
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    age: int | None = Field(None, ge=0, le=150)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tags: list[str] = Field(default_factory=list)

# Creation
user = User(id=1, name="Alice", email="alice@example.com", age=30)
print(user.model_dump())
# {'id': 1, 'name': 'Alice', 'email': 'alice@example.com', 'age': 30, 'created_at': ..., 'tags': []}

# Validation errors
from pydantic import ValidationError

try:
    bad_user = User(id="not_an_int", name="", email="invalid")
except ValidationError as e:
    print(e.errors())
    # [
    #   {'type': 'int_parsing', 'loc': ('id',), 'msg': '...'},
    #   {'type': 'string_too_short', 'loc': ('name',), 'msg': '...'},
    #   {'type': 'string_pattern_mismatch', 'loc': ('email',), 'msg': '...'},
    # ]
```

### Pydantic v2: Validators

Pydantic v2 introduced `field_validator` and `model_validator` as replacements for the v1 `@validator` and `@root_validator` decorators.

```python
from pydantic import BaseModel, field_validator, model_validator

class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    password_confirm: str

    # --- Field-level validator ---
    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Name must not be empty or whitespace")
        return v.strip().title()

    @field_validator("email")
    @classmethod
    def email_must_be_lowercase(cls, v: str) -> str:
        return v.lower()

    # --- Model-level validator (access all fields) ---
    @model_validator(mode="after")
    def passwords_must_match(self) -> "UserCreate":
        if self.password != self.password_confirm:
            raise ValueError("Passwords do not match")
        return self

    # --- Before validator (runs before standard validation) ---
    @model_validator(mode="before")
    @classmethod
    def pre_process(cls, data: dict) -> dict:
        """Example: strip whitespace from all string fields."""
        if isinstance(data, dict):
            return {k: v.strip() if isinstance(v, str) else v for k, v in data.items()}
        return data

# Usage
user = UserCreate(
    name="  alice  ",
    email="ALICE@Example.COM",
    password="secret123",
    password_confirm="secret123",
)
print(user.name)   # "Alice"
print(user.email)  # "alice@example.com"
```

### Pydantic v1 vs v2 Comparison

| Feature | Pydantic v1 | Pydantic v2 |
|---|---|---|
| **Core** | Pure Python | Rust-based `pydantic-core` (5-50x faster) |
| **Field validator** | `@validator("field")` | `@field_validator("field")` |
| **Root validator** | `@root_validator` | `@model_validator(mode="before"\|"after")` |
| **Serialization** | `.dict()`, `.json()` | `.model_dump()`, `.model_dump_json()` |
| **Parsing** | `.parse_obj()`, `.parse_raw()` | `.model_validate()`, `.model_validate_json()` |
| **Schema** | `.schema()` | `.model_json_schema()` |
| **Config** | `class Config:` inner class | `model_config = ConfigDict(...)` |
| **Strict mode** | Not available | `model_config = ConfigDict(strict=True)` |

### Settings Management with pydantic-settings

```python
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env files."""

    # Database
    database_url: str = Field("postgresql+asyncpg://localhost/mydb")
    database_pool_size: int = Field(10, ge=1, le=100)

    # Redis
    redis_url: str = Field("redis://localhost:6379/0")

    # Auth
    secret_key: str = Field(...)  # Required, must be in env
    access_token_expire_minutes: int = Field(30)

    # App
    debug: bool = Field(False)
    environment: str = Field("development")
    log_level: str = Field("INFO")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_prefix": "",         # No prefix; e.g., DATABASE_URL
        "case_sensitive": False,   # DATABASE_URL = database_url
        "extra": "ignore",
    }

# Usage:
# Reads from environment variables first, then .env file, then defaults.
settings = Settings()
print(settings.database_url)
```

### Serialization / Deserialization

```python
from pydantic import BaseModel, ConfigDict
from datetime import datetime

class Event(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,  # Allows creating from ORM objects
    )

    id: int
    title: str
    starts_at: datetime
    tags: list[str] = []

# --- Serialization ---
event = Event(id=1, title="Conference", starts_at=datetime(2026, 6, 1), tags=["tech"])

# To dict
d = event.model_dump()
# {'id': 1, 'title': 'Conference', 'starts_at': datetime(2026, 6, 1), 'tags': ['tech']}

# To dict (exclude fields)
d2 = event.model_dump(exclude={"tags"})
# {'id': 1, 'title': 'Conference', 'starts_at': datetime(2026, 6, 1)}

# To JSON string
json_str = event.model_dump_json(indent=2)

# --- Deserialization ---
# From dict
event2 = Event.model_validate({"id": 2, "title": "Meetup", "starts_at": "2026-07-01T10:00:00"})

# From JSON string
event3 = Event.model_validate_json('{"id": 3, "title": "Workshop", "starts_at": "2026-08-01T09:00:00"}')

# From ORM object (requires from_attributes=True)
class EventORM:
    id = 4
    title = "Webinar"
    starts_at = datetime(2026, 9, 1)
    tags = ["online"]

event4 = Event.model_validate(EventORM, from_attributes=True)
```

### Custom Types

```python
from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import core_schema
from typing import Any, Annotated

class Color:
    """Custom type representing an RGB color."""

    def __init__(self, r: int, g: int, b: int) -> None:
        self.r = min(max(r, 0), 255)
        self.g = min(max(g, 0), 255)
        self.b = min(max(b, 0), 255)

    def __repr__(self) -> str:
        return f"Color({self.r}, {self.g}, {self.b})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize,
                info_arg=False,
            ),
        )

    @classmethod
    def _validate(cls, value: Any) -> "Color":
        if isinstance(value, cls):
            return value
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return cls(*value)
        if isinstance(value, str) and value.startswith("#") and len(value) == 7:
            r = int(value[1:3], 16)
            g = int(value[3:5], 16)
            b = int(value[5:7], 16)
            return cls(r, g, b)
        raise ValueError(f"Cannot convert {value!r} to Color")

    @staticmethod
    def _serialize(value: "Color") -> str:
        return f"#{value.r:02x}{value.g:02x}{value.b:02x}"

class Theme(BaseModel):
    name: str
    primary: Color
    secondary: Color

theme = Theme(name="Dark", primary="#1a1a2e", secondary=[52, 73, 94])
print(theme.primary)    # Color(26, 26, 46)
print(theme.model_dump())
# {'name': 'Dark', 'primary': '#1a1a2e', 'secondary': '#34495e'}
```

---

## 8. Dependency Injection Patterns

### What Is Dependency Injection?

**Dependency Injection (DI)** is a design pattern where an object's dependencies are provided from the outside rather than created internally. This promotes:

- **Testability:** Swap real dependencies with mocks/fakes in tests.
- **Loose coupling:** Components depend on abstractions, not concrete implementations.
- **Configurability:** Change behavior by swapping implementations.

### Pattern 1: Manual DI (Constructor Injection)

```python
from typing import Protocol

# --- Define abstractions ---
class UserRepository(Protocol):
    async def get_by_id(self, user_id: int) -> dict | None: ...
    async def create(self, data: dict) -> dict: ...

class EmailService(Protocol):
    async def send(self, to: str, subject: str, body: str) -> None: ...

# --- Concrete implementations ---
class PostgresUserRepository:
    def __init__(self, db_pool) -> None:
        self._pool = db_pool

    async def get_by_id(self, user_id: int) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
            return dict(row) if row else None

    async def create(self, data: dict) -> dict:
        # ... insert into DB
        return {"id": 1, **data}

class SMTPEmailService:
    def __init__(self, smtp_host: str, smtp_port: int) -> None:
        self._host = smtp_host
        self._port = smtp_port

    async def send(self, to: str, subject: str, body: str) -> None:
        # ... send via SMTP
        print(f"Email sent to {to}")

# --- Service that receives dependencies ---
class UserService:
    def __init__(
        self,
        repo: UserRepository,
        email: EmailService,
    ) -> None:
        self._repo = repo
        self._email = email

    async def register(self, name: str, email: str) -> dict:
        user = await self._repo.create({"name": name, "email": email})
        await self._email.send(email, "Welcome!", f"Hi {name}, welcome!")
        return user

# --- Wiring (manual) ---
async def main() -> None:
    db_pool = await create_pool("postgresql://...")
    repo = PostgresUserRepository(db_pool)
    email = SMTPEmailService("smtp.example.com", 587)
    service = UserService(repo=repo, email=email)

    user = await service.register("Alice", "alice@example.com")
    print(user)
```

### Pattern 2: FastAPI's `Depends`

```python
from fastapi import FastAPI, Depends
from typing import Annotated, AsyncIterator

app = FastAPI()

# --- Dependencies as functions ---
async def get_db_session() -> AsyncIterator:
    session = AsyncSessionLocal()
    try:
        yield session
    finally:
        await session.close()

class UserRepository:
    def __init__(self, session) -> None:
        self.session = session

    async def get_by_id(self, user_id: int) -> dict | None:
        # Use self.session for DB queries
        return {"id": user_id, "name": "Alice"}

def get_user_repo(
    session: Annotated[object, Depends(get_db_session)],
) -> UserRepository:
    return UserRepository(session)

class UserService:
    def __init__(self, repo: UserRepository) -> None:
        self.repo = repo

    async def get_user(self, user_id: int) -> dict | None:
        return await self.repo.get_by_id(user_id)

def get_user_service(
    repo: Annotated[UserRepository, Depends(get_user_repo)],
) -> UserService:
    return UserService(repo)

# --- Route uses the full dependency chain ---
@app.get("/users/{user_id}")
async def read_user(
    user_id: int,
    service: Annotated[UserService, Depends(get_user_service)],
) -> dict:
    user = await service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404)
    return user

# FastAPI resolves the chain: get_db_session -> get_user_repo -> get_user_service
```

### Pattern 3: dependency-injector Library

```python
from dependency_injector import containers, providers
from dependency_injector.wiring import inject, Provide

# --- Container defines all dependencies ---
class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    # Singletons
    db_pool = providers.Singleton(
        create_pool,
        dsn=config.database_url,
    )

    redis_client = providers.Singleton(
        create_redis,
        url=config.redis_url,
    )

    # Factories (new instance each time)
    user_repository = providers.Factory(
        PostgresUserRepository,
        db_pool=db_pool,
    )

    email_service = providers.Factory(
        SMTPEmailService,
        smtp_host=config.smtp_host,
        smtp_port=config.smtp_port,
    )

    user_service = providers.Factory(
        UserService,
        repo=user_repository,
        email=email_service,
    )

# --- Usage with wiring ---
@inject
async def create_user_handler(
    name: str,
    email: str,
    service: UserService = Provide[Container.user_service],
) -> dict:
    return await service.register(name, email)

# --- Setup ---
container = Container()
container.config.from_dict({
    "database_url": "postgresql://...",
    "redis_url": "redis://...",
    "smtp_host": "smtp.example.com",
    "smtp_port": 587,
})
container.wire(modules=[__name__])

# --- Testing: override dependencies ---
def test_create_user():
    mock_repo = MockUserRepository()
    mock_email = MockEmailService()

    with container.user_repository.override(mock_repo), \
         container.email_service.override(mock_email):
        result = asyncio.run(create_user_handler("Test", "test@example.com"))
        assert result["name"] == "Test"
```

### Comparison of DI Approaches

| Approach | Pros | Cons |
|---|---|---|
| **Manual DI** | Simple, no magic, explicit | Verbose wiring code; hard to manage in large apps |
| **FastAPI Depends** | Declarative, integrates with framework, auto-cleanup with `yield` | Tied to FastAPI; dependency chains can be hard to trace |
| **dependency-injector** | Framework-agnostic, powerful overriding for tests, rich provider types | Learning curve; extra library dependency; more abstraction |

---

## 9. Project Structure for Production

### Recommended Folder Structure

```
my_service/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app creation, lifespan
│   │
│   ├── api/                       # HTTP layer (routers)
│   │   ├── __init__.py
│   │   ├── deps.py                # Shared dependencies (auth, DB session)
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py          # Aggregates all v1 routers
│   │       ├── users.py           # /api/v1/users endpoints
│   │       ├── items.py           # /api/v1/items endpoints
│   │       └── health.py          # /api/v1/health endpoint
│   │
│   ├── core/                      # App-wide configuration and utilities
│   │   ├── __init__.py
│   │   ├── config.py              # Settings (pydantic-settings)
│   │   ├── security.py            # JWT, hashing utilities
│   │   └── logging.py             # Logging configuration
│   │
│   ├── models/                    # SQLAlchemy / ORM models
│   │   ├── __init__.py
│   │   ├── base.py                # Base model class
│   │   ├── user.py
│   │   └── item.py
│   │
│   ├── schemas/                   # Pydantic schemas (request/response)
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── item.py
│   │
│   ├── services/                  # Business logic layer
│   │   ├── __init__.py
│   │   ├── user_service.py
│   │   └── item_service.py
│   │
│   ├── repositories/              # Data access layer
│   │   ├── __init__.py
│   │   ├── base.py                # Generic CRUD repository
│   │   ├── user_repo.py
│   │   └── item_repo.py
│   │
│   └── db/                        # Database connection and session
│       ├── __init__.py
│       ├── session.py             # Engine, sessionmaker
│       └── migrations/            # Alembic migrations
│           ├── env.py
│           ├── alembic.ini
│           └── versions/
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Fixtures (test DB, client, etc.)
│   ├── test_users.py
│   └── test_items.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── pyproject.toml                 # Project metadata, dependencies, tool config
├── .env.example
├── .gitignore
└── Makefile                       # Common commands
```

### Layered Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     HTTP Request                         │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Router Layer  (app/api/v1/users.py)                     │
│  - Parse request, validate input (via Pydantic)          │
│  - Call service layer                                    │
│  - Return response                                       │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Service Layer  (app/services/user_service.py)           │
│  - Business logic, orchestration                         │
│  - Calls repositories for data access                    │
│  - Does NOT know about HTTP or request/response          │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Repository Layer  (app/repositories/user_repo.py)       │
│  - Data access only (SQL queries, ORM operations)        │
│  - Does NOT contain business logic                       │
│  - Returns domain objects / dicts                        │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Database  (PostgreSQL, Redis, etc.)                     │
└──────────────────────────────────────────────────────────┘
```

### Code Examples

#### `app/core/config.py`

```python
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # App
    PROJECT_NAME: str = "my-service"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/mydb"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Auth
    SECRET_KEY: str = Field(...)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
    }

settings = Settings()
```

#### `app/db/session.py`

```python
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    echo=settings.DEBUG,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
```

#### `app/api/deps.py`

```python
from typing import Annotated, AsyncIterator
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from jose import JWTError, jwt

from app.db.session import AsyncSessionLocal
from app.core.config import settings

security = HTTPBearer()

async def get_db() -> AsyncIterator[AsyncSession]:
    """Yield a database session; ensure cleanup."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

DBSession = Annotated[AsyncSession, Depends(get_db)]

async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db: DBSession,
) -> dict:
    """Decode JWT and fetch the current user."""
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    # Fetch user from DB (simplified)
    user = await db.get(UserModel, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return user

CurrentUser = Annotated[dict, Depends(get_current_user)]
```

#### `app/schemas/user.py`

```python
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime

class UserBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., max_length=255)

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.lower().strip()

class UserUpdate(BaseModel):
    name: str | None = None
    email: str | None = None

class UserResponse(UserBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    is_active: bool
```

#### `app/repositories/user_repo.py`

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import UserModel
from app.schemas.user import UserCreate

class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, user_id: int) -> UserModel | None:
        return await self._session.get(UserModel, user_id)

    async def get_by_email(self, email: str) -> UserModel | None:
        stmt = select(UserModel).where(UserModel.email == email)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list(self, skip: int = 0, limit: int = 20) -> list[UserModel]:
        stmt = select(UserModel).offset(skip).limit(limit)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def create(self, data: UserCreate, hashed_password: str) -> UserModel:
        user = UserModel(
            name=data.name,
            email=data.email,
            hashed_password=hashed_password,
        )
        self._session.add(user)
        await self._session.commit()
        await self._session.refresh(user)
        return user
```

#### `app/services/user_service.py`

```python
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.user_repo import UserRepository
from app.schemas.user import UserCreate, UserResponse
from app.core.security import hash_password

class UserService:
    def __init__(self, session: AsyncSession) -> None:
        self._repo = UserRepository(session)

    async def register(self, data: UserCreate) -> UserResponse:
        existing = await self._repo.get_by_email(data.email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered",
            )

        hashed = hash_password(data.password)
        user = await self._repo.create(data, hashed)
        return UserResponse.model_validate(user)

    async def get_user(self, user_id: int) -> UserResponse:
        user = await self._repo.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        return UserResponse.model_validate(user)
```

#### `app/api/v1/users.py`

```python
from fastapi import APIRouter, Depends, status
from typing import Annotated

from app.api.deps import DBSession, CurrentUser
from app.schemas.user import UserCreate, UserResponse
from app.services.user_service import UserService

router = APIRouter(prefix="/users", tags=["users"])

def get_user_service(db: DBSession) -> UserService:
    return UserService(db)

UserServiceDep = Annotated[UserService, Depends(get_user_service)]

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(data: UserCreate, service: UserServiceDep) -> UserResponse:
    return await service.register(data)

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: CurrentUser) -> UserResponse:
    return UserResponse.model_validate(current_user)

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, service: UserServiceDep) -> UserResponse:
    return await service.get_user(user_id)
```

#### `app/api/v1/router.py`

```python
from fastapi import APIRouter

from app.api.v1 import users, items, health

api_router = APIRouter()
api_router.include_router(users.router)
api_router.include_router(items.router)
api_router.include_router(health.router)
```

#### `app/main.py`

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from typing import AsyncIterator

from app.api.v1.router import api_router
from app.core.config import settings
from app.db.session import engine

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup
    print(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    yield
    # Shutdown
    await engine.dispose()
    print("Shutdown complete.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

app.include_router(api_router, prefix="/api/v1")
```

---

## 10. Q&A Section

### Q1: What is the GIL and how does it affect Python performance?

**A:** The Global Interpreter Lock (GIL) is a mutex in CPython that permits only one thread to execute Python bytecode at a time. It exists because CPython's memory management uses non-thread-safe reference counting. The GIL means that **CPU-bound** multi-threaded programs gain no parallel speedup on multi-core machines -- threads take turns rather than running simultaneously. For **I/O-bound** tasks, the GIL is released during I/O operations (network, file, sleep), so threads achieve real concurrency. To bypass the GIL for CPU work, use `multiprocessing`, C extensions that release the GIL, or the experimental free-threaded mode in Python 3.13+.

---

### Q2: What is the difference between threading and multiprocessing?

**A:** `threading` runs multiple threads inside a **single process** sharing the same memory space. Due to the GIL, threads cannot execute CPU-bound Python code in true parallel -- they are **concurrent but not parallel** for CPU work, though they are effective for I/O-bound tasks. `multiprocessing` spawns **separate OS processes**, each with its own Python interpreter and GIL, enabling **true parallelism** on multiple cores. The trade-offs: multiprocessing has higher startup cost, requires data to be serialized (pickled) for inter-process communication, and uses more memory since each process has its own address space. Threading is lighter-weight but limited by the GIL.

---

### Q3: When would you use asyncio vs threading?

**A:** Use **asyncio** when you need high-concurrency I/O (thousands of simultaneous connections, API calls, or websockets). It runs in a single thread with an event loop and has minimal overhead per task. Use **threading** when you need to run blocking I/O code that is not async-compatible (e.g., legacy libraries, file system operations without async support) or when the concurrency level is moderate. Asyncio is preferred for modern Python web services because it handles scale better and avoids the complexity of thread synchronization. However, asyncio requires all code in the chain to be non-blocking; if you call a blocking function without `run_in_executor`, it blocks the entire event loop.

---

### Q4: How does FastAPI's dependency injection work?

**A:** FastAPI uses a function-based DI system via the `Depends()` marker. You declare a dependency as a function parameter annotated with `Annotated[Type, Depends(dependency_function)]`. When a request arrives, FastAPI calls the dependency function, resolves its own dependencies recursively, and injects the return value into the handler. Dependencies can be sync or async functions. Using `yield` makes a dependency a context manager -- code after `yield` runs after the response is sent (useful for cleanup like closing DB sessions). Dependencies are resolved per-request by default but can be cached within a request using the same `Depends` instance. This provides a clean separation of concerns without heavyweight DI frameworks.

---

### Q5: What are Pydantic validators and how do v1 vs v2 differ?

**A:** Pydantic validators are methods on a model that transform or validate field values during parsing. In **v1**, you use `@validator("field_name")` for field-level validation and `@root_validator` for cross-field validation. In **v2**, these are replaced by `@field_validator("field_name")` (must be a `@classmethod`) and `@model_validator(mode="before"|"after")`. The `mode="before"` variant runs before Pydantic's standard validation and receives raw data; `mode="after"` runs after validation and receives the fully constructed model instance. Pydantic v2 also replaced `.dict()` with `.model_dump()`, `.parse_obj()` with `.model_validate()`, and `class Config:` with `model_config = ConfigDict(...)`. Under the hood, v2 uses a Rust-based core that is 5-50x faster.

---

### Q6: How do you structure a production Python project?

**A:** A production Python project typically uses a **layered architecture**: Router (HTTP handling) -> Service (business logic) -> Repository (data access) -> Database. Files are organized as: `app/api/` for route handlers, `app/services/` for business logic, `app/repositories/` for data access, `app/schemas/` for Pydantic models, `app/models/` for ORM models, `app/core/` for config and utilities, and `app/db/` for database setup and migrations. Tests live in a separate `tests/` directory. Configuration uses `pydantic-settings` to load from environment variables and `.env` files. This separation ensures each layer has a single responsibility, making the code testable and maintainable.

---

### Q7: What is the event loop in asyncio?

**A:** The event loop is the central execution mechanism in asyncio. It runs in a single thread and continuously cycles through: polling for I/O events (using OS primitives like `epoll` or `kqueue`), running callbacks registered for completed I/O, advancing coroutines past their `await` points, and scheduling new tasks. When a coroutine hits an `await`, it yields control back to the event loop, which can then run other ready tasks. This cooperative multitasking approach lets a single thread handle thousands of concurrent I/O operations efficiently. You start the event loop with `asyncio.run(main())`, which creates a loop, runs the coroutine, and cleans up.

---

### Q8: How do you handle CPU-bound tasks in an async application?

**A:** CPU-bound tasks should never be run directly in the async event loop because they block it and prevent other coroutines from executing. The standard approach is to use `loop.run_in_executor()` with a `ProcessPoolExecutor` to offload CPU work to separate processes:

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

def cpu_heavy(n: int) -> float:
    return sum(i ** 0.5 for i in range(n))

async def handler() -> float:
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, cpu_heavy, 10_000_000)
    return result
```

For lighter blocking work, you can use a `ThreadPoolExecutor` (the default executor), but this still shares the GIL.

---

### Q9: What are Protocol classes in Python?

**A:** Protocol classes (from `typing`) enable **structural subtyping** (duck typing with type-checking support). A class satisfies a Protocol if it implements the required methods and attributes, without needing to explicitly inherit from it. This is similar to Go interfaces. You define a Protocol by inheriting from `typing.Protocol` and declaring the required method signatures. The `@runtime_checkable` decorator optionally enables `isinstance()` checks at runtime. Protocols are useful for defining interfaces for dependency injection and repository patterns, allowing you to swap implementations (real, mock, in-memory) without changing the code that uses them.

---

### Q10: Explain async context managers.

**A:** An async context manager is an object that defines `__aenter__` and `__aexit__` methods (both `async`), allowing it to be used with `async with`. They are used for managing async resources that require setup and teardown (e.g., database connections, HTTP sessions, locks).

```python
class AsyncDBConnection:
    async def __aenter__(self):
        self.conn = await connect_to_db()
        return self.conn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.conn.close()
        return False  # Don't suppress exceptions

# Usage:
async with AsyncDBConnection() as conn:
    await conn.execute("SELECT 1")

# Or using contextlib:
from contextlib import asynccontextmanager

@asynccontextmanager
async def get_connection():
    conn = await connect_to_db()
    try:
        yield conn
    finally:
        await conn.close()
```

---

### Q11: What is the difference between `asyncio.gather()` and `asyncio.wait()`?

**A:** `asyncio.gather()` runs awaitables concurrently and returns a **list of results in the same order** as the input. It raises the first exception by default (or returns exceptions as values with `return_exceptions=True`). `asyncio.wait()` takes a set of tasks and returns two sets: `(done, pending)`. It supports `return_when` parameter (`FIRST_COMPLETED`, `FIRST_EXCEPTION`, `ALL_COMPLETED`), giving you fine-grained control over when to proceed. Use `gather` when you need all results in order; use `wait` when you want to process results as they complete or need to handle partial completion.

---

### Q12: How does `asyncio.Semaphore` work and when would you use it?

**A:** `asyncio.Semaphore(n)` is a synchronization primitive that limits the number of concurrent coroutines that can enter a critical section to `n`. Each `async with semaphore:` acquires a slot (decrementing the counter); when the counter reaches zero, subsequent coroutines block until a slot is released. Common uses include rate-limiting API calls (e.g., max 10 concurrent requests), limiting database connection usage, and controlling access to any shared resource with a capacity limit. Unlike `asyncio.Lock` (which is a semaphore with n=1), a `Semaphore` allows multiple concurrent entries.

---

### Q13: What happens if you forget to `await` a coroutine?

**A:** If you call an async function without `await`, you get a **coroutine object** instead of the result. The coroutine never executes, and Python emits a `RuntimeWarning: coroutine 'xyz' was never awaited`. This is a common bug that can cause missing data, silent failures, or resource leaks (e.g., a DB transaction that never commits). Always use `await`, `asyncio.create_task()`, or `asyncio.gather()` to schedule coroutines. Enable Python's `-W error::RuntimeWarning` flag in development to turn these warnings into exceptions.

---

### Q14: What is the difference between `create_task()` and `await` directly?

**A:** `await coroutine()` runs the coroutine and suspends the current coroutine until it completes -- it is sequential. `asyncio.create_task(coroutine())` schedules the coroutine to run concurrently on the event loop and returns a `Task` object immediately. The current coroutine can continue to do other work. Use `create_task` when you want multiple operations to run concurrently; use `await` directly when the current step depends on the result before proceeding.

```python
# Sequential (total ~2s):
result1 = await fetch("api/a")  # waits 1s
result2 = await fetch("api/b")  # waits 1s

# Concurrent (total ~1s):
task1 = asyncio.create_task(fetch("api/a"))
task2 = asyncio.create_task(fetch("api/b"))
result1 = await task1
result2 = await task2
```

---

### Q15: How do you test async code in Python?

**A:** Use `pytest` with the `pytest-asyncio` plugin. Mark async test functions with `@pytest.mark.asyncio` and use `async def`:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_create_user():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/users/", json={
            "name": "Alice",
            "email": "alice@test.com",
            "password": "secret123",
        })
    assert response.status_code == 201
    assert response.json()["name"] == "Alice"
```

For unit tests, use `unittest.mock.AsyncMock` to mock async dependencies. For integration tests, use a test database with fixtures that set up and tear down data.

---

### Q16: What is `Annotated` and why is it preferred in modern FastAPI?

**A:** `Annotated[Type, metadata]` (from `typing`) lets you attach metadata to a type hint without changing the type itself. In FastAPI, it replaces default-value-based dependency injection:

```python
# Old style (works but overloads the default value):
@app.get("/items")
async def list_items(db: AsyncSession = Depends(get_db)):
    ...

# Modern style with Annotated (cleaner, reusable):
DBSession = Annotated[AsyncSession, Depends(get_db)]

@app.get("/items")
async def list_items(db: DBSession):
    ...
```

Benefits: the `DBSession` alias is reusable across routes; the type annotation and dependency metadata are clearly separated; function signatures remain cleaner; and IDEs provide better autocomplete because the base type is unambiguous.

---

### Q17: What is the difference between `multiprocessing.Pool` and `concurrent.futures.ProcessPoolExecutor`?

**A:** Both manage pools of worker processes, but they have different APIs and integration points. `multiprocessing.Pool` is the older API with methods like `map()`, `apply_async()`, and `starmap()`. `ProcessPoolExecutor` (from `concurrent.futures`) provides a higher-level `Executor` interface with `submit()` returning `Future` objects and `map()`. The key advantage of `ProcessPoolExecutor` is its compatibility with `asyncio` via `loop.run_in_executor()`, making it the preferred choice for async applications. It also provides a uniform API with `ThreadPoolExecutor`, so you can swap between threading and multiprocessing by changing one class.

---

### Q18: How does Pydantic's `model_config = ConfigDict(from_attributes=True)` work?

**A:** `from_attributes=True` (called `orm_mode` in Pydantic v1) tells Pydantic to read data from object attributes instead of dictionary keys. This is essential for converting ORM model instances (SQLAlchemy, Tortoise, etc.) into Pydantic response models. Without it, `Model.model_validate(orm_object)` would fail because it expects a dictionary. With it, Pydantic calls `getattr(obj, field_name)` for each field. You use it on response schemas that need to serialize database objects.

---

### Q19: What is `yield` in a FastAPI dependency and how does cleanup work?

**A:** When a FastAPI dependency function uses `yield`, it becomes a **generator-based context manager**. Code before `yield` runs at the start of the request (setup), the yielded value is injected into the handler, and code after `yield` runs after the response is sent (cleanup). This pattern is used for database sessions, file handles, and any resource that needs deterministic cleanup:

```python
async def get_db() -> AsyncIterator[AsyncSession]:
    session = AsyncSessionLocal()
    try:
        yield session         # Session is injected into the handler
    finally:
        await session.close() # Always runs after the response, even on errors
```

If the handler raises an exception, the `finally` block still runs, ensuring resources are properly released.

---

### Q20: How do you prevent blocking the asyncio event loop?

**A:** The golden rule: **never call blocking functions directly in async code**. Common blocking operations include `time.sleep()`, synchronous HTTP calls (`requests.get()`), CPU-heavy computation, and synchronous file I/O. Solutions:

1. Use async equivalents: `asyncio.sleep()`, `aiohttp`, `aiofiles`.
2. Offload blocking I/O to a thread pool: `await loop.run_in_executor(None, blocking_func)`.
3. Offload CPU work to a process pool: `await loop.run_in_executor(ProcessPoolExecutor(), cpu_func)`.
4. Use `asyncio.to_thread(blocking_func)` (Python 3.9+) as shorthand for thread offloading.
5. Enable **debug mode** (`asyncio.run(main(), debug=True)`) to detect calls that block for over 100ms.

---

### Q21: What is `TypeVar` with `bound` vs `constraints`?

**A:** A `TypeVar` with `bound=X` means the type variable can be `X` or any subclass of `X`. A `TypeVar` with constraints like `TypeVar("T", int, str)` means the type variable must be **exactly one** of the listed types (not a subclass). Example:

```python
from typing import TypeVar

# Bound: T can be any Number or subclass thereof
from numbers import Number
T_bound = TypeVar("T_bound", bound=Number)

# Constraints: T must be exactly int or str
T_constrained = TypeVar("T_constrained", int, str)

def add(a: T_bound, b: T_bound) -> T_bound:
    return a + b  # Works for int, float, complex...

def repeat(val: T_constrained, n: int) -> T_constrained:
    return val * n  # Only int or str allowed
```

Use `bound` when you want a class hierarchy; use constraints when you want specific unrelated types.

---

### Q22: What are `TaskGroup` and `ExceptionGroup` in Python 3.11+?

**A:** `asyncio.TaskGroup` provides **structured concurrency**: all tasks created within the group are guaranteed to be complete (or cancelled) when the `async with` block exits. If any task raises an exception, all other tasks in the group are cancelled, and an `ExceptionGroup` (containing all exceptions) is raised. This is safer than `gather()` because it prevents "orphan" tasks that keep running after an error.

```python
async def main():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(might_fail_1())
            tg.create_task(might_fail_2())
    except* ValueError as eg:
        # except* handles ExceptionGroups (Python 3.11+)
        for exc in eg.exceptions:
            print(f"Caught: {exc}")
```

The `except*` syntax is specifically for matching exceptions inside `ExceptionGroup`.

---

### Q23: How does aiohttp's `ClientSession` manage connection pooling?

**A:** `aiohttp.ClientSession` maintains a **connection pool** (via a `TCPConnector`) that reuses TCP connections across multiple requests to the same host. By default, it allows up to 100 connections total and 30 per host. Reusing connections avoids the overhead of TCP handshakes and TLS negotiations for every request. Best practices: create **one session** per application (or per logical group), pass it around, and close it when done. Creating a new session per request wastes the connection pool. You can customize limits:

```python
connector = aiohttp.TCPConnector(limit=200, limit_per_host=50)
async with aiohttp.ClientSession(connector=connector) as session:
    ...
```

---

### Q24: What is the difference between `@staticmethod`, `@classmethod`, and regular methods?

**A:** A **regular method** takes `self` as its first argument and operates on an instance. A **`@classmethod`** takes `cls` as its first argument and operates on the class (useful for factory methods or accessing class-level attributes). A **`@staticmethod`** takes no implicit first argument and is essentially a namespaced function -- it cannot access instance or class state. In the context of Pydantic v2, `@field_validator` must be decorated with `@classmethod` because it validates data before the instance is created. In general, use `@classmethod` for alternative constructors and `@staticmethod` for utility functions that logically belong to the class but don't need access to `self` or `cls`.

---

### Q25: How would you implement graceful shutdown in a FastAPI application?

**A:** Use FastAPI's **lifespan** context manager to handle both startup and shutdown:

```python
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI
import asyncio

# Global for tracking background tasks
background_tasks: set[asyncio.Task] = set()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup
    print("Starting up...")
    yield
    # Shutdown: cancel background tasks gracefully
    print("Shutting down...")
    for task in background_tasks:
        task.cancel()
    await asyncio.gather(*background_tasks, return_exceptions=True)
    # Close DB connections, flush caches, etc.
    await engine.dispose()
    await redis.close()
    print("Cleanup complete.")

app = FastAPI(lifespan=lifespan)
```

When the application receives a SIGTERM or SIGINT, Uvicorn triggers the shutdown sequence. The lifespan's code after `yield` executes, allowing you to cancel pending tasks, close database pools, flush log buffers, and release any other resources cleanly. For long-running background tasks, implement cooperative cancellation by checking `asyncio.current_task().cancelled()` or catching `asyncio.CancelledError`.

---

## Quick Reference Cheat Sheet

```
Concurrency Model Selection:
  I/O-bound + high concurrency  -->  asyncio
  I/O-bound + moderate concurrency --> threading
  CPU-bound                     -->  multiprocessing
  Mixed I/O + CPU               -->  asyncio + ProcessPoolExecutor

FastAPI Dependency Chain:
  get_db() --> get_repo(db) --> get_service(repo) --> route_handler(service)

Pydantic v2 Migration:
  .dict()       --> .model_dump()
  .json()       --> .model_dump_json()
  .parse_obj()  --> .model_validate()
  @validator    --> @field_validator + @classmethod
  @root_validator --> @model_validator(mode="before"|"after")
  class Config: --> model_config = ConfigDict(...)

Type Hint Quick Reference:
  str | None          = Optional[str]
  int | str           = Union[int, str]
  list[int]           = List[int]       (Python 3.9+)
  dict[str, Any]      = Dict[str, Any]  (Python 3.9+)
  tuple[int, ...]     = variable-length tuple of ints
  Callable[[int], str] = function (int) -> str
  Annotated[T, meta]  = T with attached metadata
```
