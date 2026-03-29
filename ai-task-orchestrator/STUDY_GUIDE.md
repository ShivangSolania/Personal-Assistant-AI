# 📚 Study Guide — AI Task Orchestration System

A comprehensive guide to every module, library, and function used in this project.

---

## Table of Contents

1. [External Libraries](#1-external-libraries)
2. [Module: `utils/logger.py`](#2-module-utilsloggerpy)
3. [Module: `utils/parser.py`](#3-module-utilsparserpy)
4. [Module: `mcp/registry.py`](#4-module-mcpregistrypy)
5. [Module: `mcp/client.py`](#5-module-mcpclientpy)
6. [Module: `mcp/tools/*`](#6-module-mcptools)
7. [Module: `memory/memory_store.py`](#7-module-memorymemory_storepy)
8. [Module: `agents/planner.py`](#8-module-agentsplannerpy)
9. [Module: `agents/executor.py`](#9-module-agentsexecutorpy)
10. [Module: `graph/workflow.py`](#10-module-graphworkflowpy)
11. [Module: `main.py`](#11-module-mainpy)
12. [Key Python Concepts Used](#12-key-python-concepts-used)

---

## 1. External Libraries

### 🔗 LangGraph (`langgraph`)

LangGraph is a library for building **stateful, multi-step AI workflows** as directed graphs.

| Concept | What It Does |
|---------|-------------|
| `StateGraph` | Creates a graph where each node is a function that reads/writes shared state |
| `END` | Special constant marking the terminal node of the graph |
| `add_node()` | Registers a function as a named node in the graph |
| `add_edge()` | Creates a fixed connection between two nodes |
| `add_conditional_edges()` | Creates a branch — a function decides which node to go to next |
| `set_entry_point()` | Declares which node runs first |
| `compile()` | Freezes the graph into a runnable object |
| `ainvoke()` | Runs the compiled graph asynchronously with an initial state |

**Why we use it:** It gives us a visual, debuggable way to orchestrate Planner → Executor → Memory → Output with looping.

---

### 🔗 LangChain (`langchain`, `langchain-openai`, `langchain-core`)

LangChain is a framework for building LLM-powered applications.

| Class / Function | Module | Purpose |
|-----------------|--------|---------|
| `ChatOpenAI` | `langchain_openai` | Wrapper for any OpenAI-compatible chat API (GPT, Qwen, Llama, etc.) |
| `SystemMessage` | `langchain_core.messages` | A message with the "system" role — sets behaviour/instructions for the LLM |
| `HumanMessage` | `langchain_core.messages` | A message with the "user" role — the actual query |
| `ainvoke(messages)` | method on `ChatOpenAI` | Sends messages to the LLM asynchronously and returns the response |

**Key parameters for `ChatOpenAI`:**
```python
ChatOpenAI(
    model="gpt-4o-mini",       # Model name
    openai_api_base="https://...",  # API endpoint URL
    openai_api_key="sk-...",   # Your API key
    temperature=0.2,           # 0 = deterministic, 1 = creative
    max_tokens=2048,           # Max response length
)
```

---

### 🔗 Pydantic (`pydantic`)

Pydantic is a **data validation** library. You define a class with type hints, and Pydantic enforces those types at runtime.

| Class / Function | Purpose |
|-----------------|---------|
| `BaseModel` | Base class — any subclass automatically validates its fields |
| `Field()` | Customize a field (default values, descriptions, aliases) |
| `model_validate(data)` | Parse a dict into a validated Pydantic model instance |
| `model_dump()` | Convert a model instance back to a plain dict |

**Example from our project:**
```python
class Task(BaseModel):
    id: int                    # must be an integer
    description: str           # must be a string
    tool: str
    tool_input: dict = Field(default_factory=dict)  # defaults to {}
```

---

### 🔗 httpx

An **async-capable HTTP client** (modern alternative to `requests`).

| Function / Class | Purpose |
|-----------------|---------|
| `httpx.AsyncClient()` | Creates an async HTTP session |
| `client.get(url)` | Async GET request |
| `client.post(url, json=body)` | Async POST with JSON body |
| `timeout` parameter | Seconds before the request times out |
| `follow_redirects=True` | Automatically follow 3xx redirects |

---

### 🔗 python-dotenv

Loads environment variables from a `.env` file into `os.environ`.

```python
from dotenv import load_dotenv
load_dotenv()  # reads .env file in current directory
value = os.getenv("OPENAI_API_KEY")  # now available
```

---

### 🔗 FAISS (`faiss-cpu`)

Facebook AI Similarity Search — a library for **fast vector similarity search**.

| Function / Class | Purpose |
|-----------------|---------|
| `faiss.IndexFlatL2(dim)` | Creates an index using L2 (Euclidean) distance |
| `index.add(vectors)` | Add vectors to the index |
| `index.search(query_vec, k)` | Find the `k` nearest vectors |
| `index.ntotal` | Number of vectors currently stored |
| `index.reset()` | Clear all vectors |

**Why we use it:** For semantic memory search — finding past conversations similar to the current query.

---

### 🔗 Python Standard Library Modules

| Module | Functions Used | Purpose |
|--------|---------------|---------|
| `ast` | `ast.parse()`, `ast.Expression`, `ast.BinOp`, etc. | Parse math expressions into an Abstract Syntax Tree for safe evaluation |
| `asyncio` | `asyncio.run()` | Run async functions from synchronous code |
| `json` | `json.loads()`, `json.dumps()` | Parse and serialize JSON strings |
| `logging` | `logging.getLogger()`, `StreamHandler`, `FileHandler` | Python's built-in logging framework |
| `os` | `os.getenv()`, `os.makedirs()`, `os.path.join()` | Environment variables, file system operations |
| `pathlib` | `Path.resolve()`, `Path.exists()`, `Path.read_text()` | Modern file path manipulation |
| `re` | `re.search()` | Regular expression pattern matching |
| `importlib` | `importlib.import_module()` | Dynamically import Python modules by name at runtime |
| `pkgutil` | `pkgutil.iter_modules()` | Discover all sub-modules in a package directory |
| `math` | `math.sqrt`, `math.sin`, `math.log`, `math.pi` | Mathematical functions for the calculator tool |
| `operator` | `operator.add`, `operator.mul`, etc. | Function versions of Python operators (+, *, etc.) |
| `time` | `time.perf_counter()`, `time.time()` | High-precision timing and timestamps |
| `dataclasses` | `@dataclass`, `field()` | Lightweight classes with auto-generated `__init__` |
| `sys` | `sys.path`, `sys.stdout` | System path manipulation, standard output |

---

## 2. Module: `utils/logger.py`

**Purpose:** Centralized, colored logging for the entire project.

### Classes & Functions

| Name | Type | Description |
|------|------|-------------|
| `_ColourFormatter` | Class | Custom `logging.Formatter` that wraps log-level names in ANSI color codes |
| `_ColourFormatter.format(record)` | Method | Called by Python's logging system for every log message — adds colors |
| `get_logger(name)` | Function | Factory that creates and configures a logger with console + file handlers |

### How `get_logger` works:
```
1. Creates a logger with the given name
2. Sets log level from LOG_LEVEL env var (default: INFO)
3. Adds a console handler with colored output
4. Adds a file handler that writes to logs/orchestrator_YYYYMMDD.log
5. Returns the configured logger (reuses if already set up)
```

---

## 3. Module: `utils/parser.py`

**Purpose:** Pydantic models for structured agent I/O, plus JSON extraction from LLM text.

### Models

| Model | Fields | Used By |
|-------|--------|---------|
| `Task` | `id`, `description`, `tool`, `tool_input` | Planner output |
| `PlannerOutput` | `tasks: list[Task]` | Planner agent |
| `ExecutorOutput` | `task_id`, `status`, `result`, `error` | Executor agent |

### Functions

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `_extract_json(text)` | Raw LLM text | Clean JSON string | Strips markdown fences and finds the first JSON object |
| `parse_planner_output(raw)` | Raw LLM text | `PlannerOutput` | Extracts JSON → validates with Pydantic |
| `parse_executor_output(raw)` | Raw LLM text | `ExecutorOutput` | Extracts JSON → validates with Pydantic |

### Why `_extract_json` is needed:
LLMs often wrap JSON inside markdown code fences like ` ```json ... ``` `. This function strips that wrapping so `json.loads()` can parse it.

---

## 4. Module: `mcp/registry.py`

**Purpose:** Central registry where all MCP tools are stored and discovered.

### Class: `ToolRegistry`

| Method | Description |
|--------|-------------|
| `__init__()` | Creates an empty `_tools` dictionary |
| `register(name, description, input_schema, output_schema, handler)` | Stores a tool definition in the registry |
| `get(name)` | Looks up a tool by name, returns `None` if not found |
| `list_tools()` | Returns metadata (no handler) for all tools — used by the Planner |
| `tool_names()` | Returns just the names as a list |
| `auto_discover(package_path)` | Scans `mcp/tools/` for `.py` files, imports each, calls its `register()` function |

### How auto-discovery works:
```
1. pkgutil.iter_modules() lists all .py files in mcp/tools/
2. For each file, importlib.import_module() loads it
3. If the module has a register() function, we call it with our registry
4. The tool self-registers with its name, schema, and handler
```

---

## 5. Module: `mcp/client.py`

**Purpose:** Bridge between agents and tools — formats MCP requests, dispatches, returns responses.

### Class: `MCPClient`

| Method | Description |
|--------|-------------|
| `__init__(registry)` | Stores a reference to the `ToolRegistry` |
| `call_tool(tool_name, tool_input)` | Async method: looks up tool → calls handler → returns envelope |
| `list_available_tools()` | Returns tool metadata for the Planner to see |
| `_envelope(tool, status, ...)` | Static helper that builds a standardized response dict |

### Response envelope format:
```python
{
    "tool": "calculator",
    "status": "success",       # or "error"
    "result": {...},           # present on success
    "error": "...",            # present on error
    "elapsed_ms": 1.2          # execution time
}
```

---

## 6. Module: `mcp/tools/*`

Each tool file follows the **same pattern**:

```python
TOOL_NAME = "..."           # Unique identifier
TOOL_DESCRIPTION = "..."    # Shown to the Planner LLM
INPUT_SCHEMA = {...}        # JSON Schema for input
OUTPUT_SCHEMA = {...}       # JSON Schema for output

async def handler(**kwargs):    # The actual logic
    ...
    return {result}

def register(registry):     # Called by auto-discovery
    registry.register(name=TOOL_NAME, ..., handler=handler)
```

### Tool Details

#### `calculator.py` — Safe Math Evaluator
- Uses `ast.parse(expression, mode="eval")` to convert math strings into an AST
- The `_safe_eval()` function walks the tree recursively
- Only whitelisted operators (`+`, `-`, `*`, `/`, `**`, `%`) and functions (`sqrt`, `sin`, `log`, etc.) are allowed
- **Why not `eval()`?** → `eval()` can execute arbitrary Python code (security risk)

#### `web_search.py` — Simulated Web Search
- Returns hardcoded simulated results for demonstration
- In production: swap the handler body with a real API call (SerpAPI, Tavily, etc.)

#### `file_reader.py` — Local File Reader
- Uses `pathlib.Path` for safe path resolution
- Validates: file exists, is actually a file (not directory), size < 1 MB
- Reads content with UTF-8 encoding

#### `custom_api.py` — Generic HTTP Caller
- Uses `httpx.AsyncClient` for non-blocking HTTP requests
- Supports GET and POST with optional headers and JSON body
- Truncates responses > 10,000 characters to avoid memory issues

---

## 7. Module: `memory/memory_store.py`

**Purpose:** Store conversation history and tool outputs for context-aware planning.

### Class Hierarchy

```
MemoryManager (facade)
    ├── InMemoryStore      (default)
    └── VectorMemoryStore  (optional, FAISS-backed)
```

### `MemoryEntry` (dataclass)
Fields: `role`, `content`, `metadata`, `timestamp`

### `InMemoryStore`

| Method | Description |
|--------|-------------|
| `add(role, content)` | Appends a new entry, trims if over `max_entries` |
| `get_history(last_n)` | Returns the last N entries as dicts |
| `search(query, top_k)` | Keyword-based search: counts word overlaps, returns top matches |
| `clear()` | Empties the store |

### `VectorMemoryStore`

| Method | Description |
|--------|-------------|
| `add(role, content)` | Stores in both the list AND the FAISS index |
| `search(query, top_k)` | Uses FAISS vector similarity to find semantically similar entries |
| `_embed(text)` | Simple bag-of-characters embedding (replace with a real model in production) |

### `MemoryManager`
- Factory that reads `MEMORY_BACKEND` env var (`in_memory` or `faiss`)
- Delegates all calls to the chosen backend
- **Pattern used:** Facade pattern — one simple interface hiding multiple implementations

---

## 8. Module: `agents/planner.py`

**Purpose:** Uses an LLM to decompose a user query into a structured plan.

### Class: `PlannerAgent`

| Method | Description |
|--------|-------------|
| `__init__(llm, tools_metadata)` | Stores the LLM client and tool descriptions |
| `_build_tools_description()` | Formats tool metadata into a readable string for the system prompt |
| `plan(query, context)` | Sends the query to the LLM → parses response → returns `PlannerOutput` |

### How `plan()` works:
```
1. Build system prompt with:
   - Available tools and their schemas
   - Previous conversation context
   - JSON output format instructions
2. Send [SystemMessage, HumanMessage] to LLM via ainvoke()
3. LLM returns raw text containing JSON
4. parse_planner_output() extracts and validates the JSON
5. Returns a PlannerOutput with a list of Task objects
```

---

## 9. Module: `agents/executor.py`

**Purpose:** Executes a single task by calling the appropriate MCP tool,
then uses the LLM to interpret the result.

### Class: `ExecutorAgent`

| Method | Description |
|--------|-------------|
| `__init__(llm, mcp_client)` | Stores the LLM and MCP client |
| `execute(task)` | Calls tool → feeds raw result to LLM → returns `ExecutorOutput` |

### How `execute()` works:
```
1. Extract tool_name and tool_input from the task dict
2. If tool is "no_tool", skip the MCP call
3. Otherwise, call mcp_client.call_tool(name, input)
4. Build a system prompt with the task details + tool result
5. Send to LLM via ainvoke() for human-readable interpretation
6. Parse the LLM response into ExecutorOutput
7. On any error, return ExecutorOutput with status="failed"
```

---

## 10. Module: `graph/workflow.py`

**Purpose:** The LangGraph state machine that orchestrates everything.

### State: `OrchestratorState` (TypedDict)

| Field | Type | Purpose |
|-------|------|---------|
| `user_input` | `str` | The user's query |
| `plan` | `dict` | The Planner's structured plan |
| `current_task_index` | `int` | Which task the Executor is on |
| `results` | `list[dict]` | Accumulated executor results |
| `memory_context` | `str` | Context string from memory |
| `final_output` | `str` | The formatted answer |

### Node Functions

| Node | What It Does |
|------|-------------|
| `user_input_node` | Initializes `current_task_index=0` and `results=[]` |
| `planner_node` | Calls `PlannerAgent.plan()` → stores the plan in state |
| `executor_node` | Picks task at `current_task_index` → calls `ExecutorAgent.execute()` → increments index |
| `memory_update_node` | Stores query + results in `MemoryManager` → builds context string |
| `final_output_node` | Formats all results into a user-friendly string with ✅/❌ icons |

### Routing Function

```python
def _tasks_remaining(state) -> str:
    if current_task_index < total_tasks:
        return "continue"   # → loops back to executor_node
    return "done"           # → proceeds to memory_update_node
```

### Graph Flow
```
Entry → user_input → planner → executor ─┐
                                  ↑       │
                                  └───────┘ (while tasks remain)
                                          │
                                          ↓ (all done)
                              memory_update → final_output → END
```

### `build_workflow()` Function
Takes planner, executor, memory → constructs nodes → wires edges → compiles → returns runnable graph.

---

## 11. Module: `main.py`

**Purpose:** Entry point — bootstraps all components and runs the interactive REPL.

### Functions

| Function | Description |
|----------|-------------|
| `_init_llm()` | Reads env vars → creates `ChatOpenAI` instance |
| `_init_tools()` | Creates `ToolRegistry` → auto-discovers → creates `MCPClient` |
| `run_interactive()` | Async REPL: reads input → invokes workflow → prints output |
| `main()` | Synchronous entry: calls `asyncio.run(run_interactive())` |

### REPL Loop:
```
1. load_dotenv() — load .env file
2. Initialize: LLM, tools, memory, planner, executor, workflow
3. Loop:
   a. Read user input
   b. Build initial state dict
   c. Call workflow.ainvoke(state) — runs the entire graph
   d. Extract final_output from the returned state
   e. Print the result
   f. Store in memory
```

---

## 12. Key Python Concepts Used

### `async` / `await`
Allows non-blocking I/O. When you `await` an API call, Python can do other work while waiting for the response. All our tool handlers and agent methods are `async`.

### `TypedDict`
A dictionary with predefined key types. Used for the LangGraph state so we get type safety without a full class.

### `dataclass`
Auto-generates `__init__`, `__repr__`, etc. from field annotations. Used for `MemoryEntry`.

### `__future__.annotations`
Makes all type hints strings by default (deferred evaluation). Prevents issues with forward references and allows `dict[str, Any]` syntax on older Python.

### f-strings
`f"Result: {value}"` — Python's string interpolation syntax, used extensively for logging and output.

### Context Managers (`async with`)
`async with httpx.AsyncClient() as client:` — ensures the HTTP client is properly closed when done.

### Factory Pattern
`MemoryManager` picks `InMemoryStore` or `VectorMemoryStore` based on configuration — the caller doesn't need to know which one.

### Self-Registration Pattern
Each tool module has a `register()` function. The registry calls it during auto-discovery, and the tool registers itself. This makes adding new tools a zero-config operation.
