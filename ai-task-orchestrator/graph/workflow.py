"""
graph/workflow.py
─────────────────
LangGraph state-graph that orchestrates:
    user_input → planner → (executor loop) → memory_update → final_output
"""

from __future__ import annotations #for current versions of python it is not needed but it is good for backward compatibility

import json
from typing import Any, Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent
from memory.memory_store import MemoryManager
from utils.logger import get_logger
from utils.parser import PlannerOutput, ExecutorOutput

log = get_logger(__name__)


# ── Graph State ───────────────────────────────────────────

class OrchestratorState(TypedDict, total=False):
    user_input: str
    plan: dict[str, Any] | None
    current_task_index: int
    results: list[dict[str, Any]]
    memory_context: str
    final_output: str


# ── Node Implementations ─────────────────────────────────

#Factory Functions, they return the node function, and they are used to create the nodes & reuseability and etc,
# if you dont use factory functions then you have to pass the dependencies to the node function every time you create it, you can do w/o it but it wont be that resuable.
# for example if you dont use factory functions then you have to pass the planner and executor and memory to the node function every time you create it, 
# but with factory functions you just pass the dependencies once when you create the factory function, and then you can use it to create the nodes 
# without passing the dependencies every time

def _build_user_input_node():
    """PassThrough — state already has user_input set by the caller."""

    async def user_input_node(state: OrchestratorState) -> OrchestratorState:
        log.info("━━ user_input_node ━━  query=%s", state["user_input"][:100])
        return {
            "current_task_index": 0,
            "results": [],
        }

    return user_input_node


def _build_planner_node(planner: PlannerAgent):
    async def planner_node(state: OrchestratorState) -> OrchestratorState:
        log.info("━━ planner_node ━━")
        context = state.get("memory_context", "")
        plan: PlannerOutput = await planner.plan(state["user_input"], context)
        log.info("Plan created with %d task(s)", len(plan.tasks))
        return {
            "plan": plan.model_dump(),
            "current_task_index": 0,
            "results": [],
        }

    return planner_node


def _build_executor_node(executor: ExecutorAgent):
    async def executor_node(state: OrchestratorState) -> OrchestratorState:
        plan_data = state["plan"]
        idx = state["current_task_index"]
        tasks = plan_data["tasks"]

        if idx >= len(tasks):
            log.warning("executor_node called but no tasks left")
            return {}

        task = tasks[idx]
        log.info("━━ executor_node ━━  task #%d / %d", idx + 1, len(tasks))

        result: ExecutorOutput = await executor.execute(task)
        results = list(state.get("results", []))
        results.append(result.model_dump())

        return {
            "results": results,
            "current_task_index": idx + 1,
        }

    return executor_node


def _build_memory_update_node(memory: MemoryManager):
    async def memory_update_node(state: OrchestratorState) -> OrchestratorState:
        log.info("━━ memory_update_node ━━")
        # Store user query
        memory.add("user", state["user_input"])

        # Store each result
        for r in state.get("results", []):
            memory.add(
                "tool",
                json.dumps(r, default=str),
                task_id=r.get("task_id"),
            )

        # Build context for future queries
        history = memory.get_history(last_n=10)
        context_str = "\n".join(
            f"[{h['role']}] {h['content'][:200]}" for h in history
        )
        return {"memory_context": context_str}

    return memory_update_node


def _build_final_output_node():
    async def final_output_node(state: OrchestratorState) -> OrchestratorState:
        log.info("━━ final_output_node ━━")
        results = state.get("results", [])

        if not results:
            return {"final_output": "No results were produced."}

        lines: list[str] = []
        for r in results:
            status_icon = "✅" if r.get("status") == "completed" else "❌"
            lines.append(
                f"{status_icon} Task #{r.get('task_id', '?')}: {r.get('result', r.get('error', 'N/A'))}"
            )

        final = "\n".join(lines)
        log.info("Final output:\n%s", final)
        return {"final_output": final}

    return final_output_node


# ── Conditional Router ────────────────────────────────────

def _tasks_remaining(state: OrchestratorState) -> str:
    """Return 'continue' if there are un-executed tasks, else 'done'."""
    plan = state.get("plan")
    if plan is None:
        return "done"
    idx = state.get("current_task_index", 0)
    total = len(plan.get("tasks", []))
    if idx < total:
        return "continue"
    return "done"


# ── Graph Builder ─────────────────────────────────────────

def build_workflow(
    planner: PlannerAgent,
    executor: ExecutorAgent,
    memory: MemoryManager,
) -> Any:
    """Construct and compile the LangGraph orchestration graph."""

    graph = StateGraph(OrchestratorState)

    # Add nodes
    graph.add_node("user_input",    _build_user_input_node())
    graph.add_node("planner",       _build_planner_node(planner))
    graph.add_node("executor",      _build_executor_node(executor))
    graph.add_node("memory_update", _build_memory_update_node(memory))
    graph.add_node("final_output",  _build_final_output_node())

    # Edges
    graph.set_entry_point("user_input")
    graph.add_edge("user_input", "planner")
    graph.add_edge("planner", "executor")

    # Conditional: loop executor until all tasks done
    graph.add_conditional_edges(
        "executor",
        _tasks_remaining,
        {
            "continue": "executor",
            "done": "memory_update",
        },
    )

    graph.add_edge("memory_update", "final_output")
    graph.add_edge("final_output", END)

    compiled = graph.compile()
    log.info("LangGraph workflow compiled successfully")
    return compiled
