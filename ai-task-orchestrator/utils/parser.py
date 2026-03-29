"""
utils/parser.py
───────────────
Pydantic models and helpers for structured I/O between agents.
"""

from __future__ import annotations #for current versions of python it is not needed but it is good for backward compatibility

import json
import re
from typing import Any, Optional #to use any operator

from pydantic import BaseModel, Field

from utils.logger import get_logger

log = get_logger(__name__)


# ── Pydantic Models ──────────────────────────────────────────

class Task(BaseModel):
    """A single subtask produced by the Planner."""
    id: int
    description: str
    tool: str
    tool_input: dict[str, Any] = Field(default_factory=dict)


class PlannerOutput(BaseModel):
    """The full plan returned by the Planner agent."""
    tasks: list[Task]


class ExecutorOutput(BaseModel):
    """Result of executing a single task."""
    task_id: int
    status: str = "completed" #completed | failed
    result: str = ""
    error: Optional[str] = None


# ── Parsing Helpers ──────────────────────────────────────────
#they are private functions because they start with _

def _extract_json(text: str) -> str:
    """Pull the first JSON object/array out of an LLM response."""
    # Try fenced code block first
    match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text) #strict regex for good json match
    if match:
        return match.group(1).strip()
    # Fall back to first { … }
    match = re.search(r"(\{[\s\S]*\})", text) #less strict regex for good json match
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_planner_output(raw: str) -> PlannerOutput:
    """Validate a Planner response into a PlannerOutput model."""
    json_str = _extract_json(raw)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        log.error("Planner returned invalid JSON: %s", exc)
        raise ValueError(f"Cannot parse planner output: {exc}") from exc

    plan = PlannerOutput.model_validate(data)
    log.info("Parsed plan with %d task(s)", len(plan.tasks))
    return plan


def parse_executor_output(raw: str) -> ExecutorOutput:
    """Validate an Executor response into an ExecutorOutput model."""
    json_str = _extract_json(raw)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        log.error("Executor returned invalid JSON: %s", exc)
        raise ValueError(f"Cannot parse executor output: {exc}") from exc

    return ExecutorOutput.model_validate(data)
