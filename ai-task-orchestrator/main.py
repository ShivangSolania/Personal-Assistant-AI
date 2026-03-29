"""
main.py
───────
Entry point for the AI Task Orchestration System.

Usage:
    python main.py
"""

from __future__ import annotations

import asyncio
import os
import sys

# Ensure project root is on sys.path for intra-package imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from mcp.registry import ToolRegistry
from mcp.client import MCPClient
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent
from memory.memory_store import MemoryManager
from graph.workflow import build_workflow
from utils.logger import get_logger

log = get_logger("main")

# ── Banner ────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║           🤖  AI Task Orchestration System  🤖              ║
║  ─────────────────────────────────────────────────────────  ║
║   Planner → Executor → MCP Tools │ Powered by LangGraph    ║
╚══════════════════════════════════════════════════════════════╝
"""


def _init_llm() -> ChatOpenAI:
    """Initialise the LLM (Qwen / OpenAI-compatible endpoint)."""
    base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")

    if not api_key:
        log.warning(
            "OPENAI_API_KEY not set — the system will fail on LLM calls. "
            "Copy .env.example → .env and add your key."
        )

    llm = ChatOpenAI(
        model=model,
        openai_api_base=base_url,
        openai_api_key=api_key,
        temperature=0.2,
        max_tokens=2048,
    )
    log.info("LLM ready: model=%s  base=%s", model, base_url)
    return llm


def _init_tools() -> tuple[ToolRegistry, MCPClient]:
    """Auto-discover MCP tools and build the client."""
    registry = ToolRegistry()
    registry.auto_discover()
    log.info("Registered tools: %s", registry.tool_names())
    client = MCPClient(registry)
    return registry, client


async def run_interactive() -> None:
    """Main interactive REPL loop."""
    load_dotenv()

    print(BANNER)

    # ── Bootstrap components ──────────────────────────────
    llm = _init_llm()
    registry, mcp_client = _init_tools()
    memory = MemoryManager()

    planner = PlannerAgent(llm, mcp_client.list_available_tools())
    executor = ExecutorAgent(llm, mcp_client)
    workflow = build_workflow(planner, executor, memory)

    print("Type your query below (or 'quit' / 'exit' to leave).\n")

    while True:
        try:
            user_input = input("🟢 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye! 👋")
            break

        # Build initial state
        initial_state = {
            "user_input": user_input,
            "plan": None,
            "current_task_index": 0,
            "results": [],
            "memory_context": "",
            "final_output": "",
        }

        print("\n⏳ Processing …\n")
        try:
            final_state = await workflow.ainvoke(initial_state)
            output = final_state.get("final_output", "No output produced.")

            # Store assistant reply in memory
            memory.add("assistant", output)

            print(f"\n🤖 Assistant:\n{output}\n")
            print("─" * 60)

        except Exception as exc:
            log.error("Workflow error: %s", exc, exc_info=True)
            print(f"\n❌ Error: {exc}\n")


# ── CLI entry ─────────────────────────────────────────────

def main() -> None:
    asyncio.run(run_interactive())


if __name__ == "__main__":
    main()
