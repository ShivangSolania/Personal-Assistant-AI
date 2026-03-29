# AI Task Orchestration System

A modular, production-ready AI assistant that decomposes user queries into subtasks, routes them dynamically through a multi-agent pipeline, and executes them via MCP-compatible tools — all orchestrated by **LangGraph**.

---

## Architecture

```
User Query
    │
    ▼
┌──────────────┐
│  user_input   │  Capture query
└──────┬───────┘
       ▼
┌──────────────┐
│   planner     │  LLM decomposes query into structured JSON plan
└──────┬───────┘
       ▼
┌──────────────┐
│   executor    │◄─── loops until all tasks complete
│   (MCP call)  │     calls tools via MCPClient
└──────┬───────┘
       ▼
┌──────────────┐
│ memory_update │  Store conversation + tool outputs
└──────┬───────┘
       ▼
┌──────────────┐
│ final_output  │  Format and return results
└──────────────┘
```

### Agents

| Agent | Role |
|-------|------|
| **Planner** | Uses LLM to break a user query into numbered subtasks, each mapped to a tool |
| **Executor** | Takes one task, calls the MCP tool, and interprets the result with the LLM |

### LangGraph Workflow

The orchestration graph has **conditional looping**: after each executor step it checks whether tasks remain. If yes → loop back to executor. If no → proceed to memory update and final output.

---

## Model Context Protocol (MCP)

Every tool is registered as an **MCP-compatible function** with:

| Field | Purpose |
|-------|---------|
| `name` | Unique identifier |
| `description` | What the tool does (shown to Planner) |
| `input_schema` | JSON Schema for the tool's input |
| `output_schema` | JSON Schema for the tool's output |
| `handler` | Async Python function that does the work |

### Built-in Tools

| Tool | Description |
|------|-------------|
| `web_search` | Simulated web search (swap in SerpAPI / Tavily for production) |
| `calculator` | Safe AST-based math evaluator |
| `file_reader` | Reads local files with size/path validation |
| `custom_api` | Generic HTTP GET/POST caller via `httpx` |

### Tool Auto-Discovery

Drop a new `.py` file into `mcp/tools/` with a `register(registry)` function and it will be picked up automatically on startup — no config changes needed.

---

## Memory System

Two backends available (set via `MEMORY_BACKEND` env var):

| Backend | Description |
|---------|-------------|
| `in_memory` | List-backed store with keyword search (default) |
| `faiss` | FAISS vector index for semantic similarity search |

Memory stores conversation history and tool outputs, enabling **context-aware planning** across multiple queries.

---

## 💬 Example Queries

```
You: What is 25 * 47 and also search for latest Python news?

Assistant:
Task #1: 25 × 47 = 1175
Task #2: Found 3 results about "latest Python news"…
```

```
You: Read the contents of requirements.txt

Assistant:
Task #1: File contains 9 dependencies including langgraph, langchain…
```

---

## Project Structure

```
ai-task-orchestrator/
├── agents/
│   ├── planner.py          # Planner Agent (LLM query decomposition)
│   └── executor.py         # Executor Agent (MCP tool calling + LLM interpretation)
├── mcp/
│   ├── registry.py         # Tool registry with auto-discovery
│   ├── client.py           # MCP client (request/response handling)
│   └── tools/
│       ├── web_search.py   # Web search tool
│       ├── calculator.py   # Math evaluator tool
│       ├── file_reader.py  # File reader tool
│       └── custom_api.py   # HTTP API caller tool
├── graph/
│   └── workflow.py         # LangGraph state machine
├── memory/
│   └── memory_store.py     # Memory backends (InMemory + FAISS)
├── utils/
│   ├── logger.py           # Colored logging
│   └── parser.py           # Pydantic models + JSON parsing
├── main.py                 # Entry point (interactive REPL)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Future Improvements

- **Real web search** — integrate SerpAPI, Brave, or Tavily
- **Streaming output** — stream executor results as they complete
- **Parallel execution** — run independent tasks concurrently
- **Persistent memory** — SQLite or Redis-backed conversation store
- **Authentication** — API key management for external tool calls
- **REST API wrapper** — serve the orchestrator via FastAPI
- **Tool chaining** — allow one tool's output to feed into another
- **Human-in-the-loop** — pause for user confirmation on sensitive actions
