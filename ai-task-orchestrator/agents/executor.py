"""
agents/executor.py
──────────────────
Executor Agent — takes a single task from the plan, calls the appropriate
MCP tool, and returns a structured result.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from mcp.client import MCPClient
from utils.logger import get_logger
from utils.parser import ExecutorOutput

log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a Task Executor AI.  You receive a single task to execute.

## Task
{task_json}

## Tool Result
{tool_result}

## Instructions
Interpret the tool result (or lack thereof) and produce a concise, helpful
answer for the user.  Return ONLY valid JSON — no markdown fences.

## Output Format
{{
  "task_id": {task_id},
  "status": "completed",
  "result": "Your concise answer here"
}}

If the tool failed or the result is empty, set status to "failed" and describe
the problem in the "result" field.
"""


class ExecutorAgent:
    """Execute a single task: call MCP tool → interpret result via LLM."""

    def __init__(self, llm: ChatOpenAI, mcp_client: MCPClient) -> None:
        self._llm = llm
        self._mcp = mcp_client

    async def execute(self, task: dict[str, Any]) -> ExecutorOutput:
        """
        Execute *task* dict (from PlannerOutput) and return an ExecutorOutput.

        Steps
        -----
        1. Call the designated MCP tool.
        2. Feed the raw result into the LLM for interpretation.
        3. Parse and return the structured output.
        """
        task_id = task["id"]
        tool_name = task.get("tool", "no_tool")
        tool_input = task.get("tool_input", {})
        description = task.get("description", "")

        log.info("Executor running task #%d — tool=%s", task_id, tool_name)

        # ── Step 1: MCP tool call ─────────────────────────
        if tool_name == "no_tool":
            tool_result_str = "(No tool required — provide a direct answer.)"
        else:
            tool_response = await self._mcp.call_tool(tool_name, tool_input)
            tool_result_str = json.dumps(tool_response, indent=2, default=str)

        # ── Step 2: LLM interpretation ───────────────────
        system = _SYSTEM_PROMPT.format(
            task_json=json.dumps(task, indent=2),
            tool_result=tool_result_str,
            task_id=task_id,
        )

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"Execute task #{task_id}: {description}"),
        ]

        try:
            response = await self._llm.ainvoke(messages)
            raw = response.content
            log.debug("Executor raw response:\n%s", raw)

            # Try to parse the LLM output as JSON
            from utils.parser import parse_executor_output
            result = parse_executor_output(raw)
            return result

        except Exception as exc:
            log.error("Executor failed on task #%d: %s", task_id, exc)
            return ExecutorOutput(
                task_id=task_id,
                status="failed",
                result="",
                error=str(exc),
            )
