"""
agents/planner.py
─────────────────
Planner Agent — decomposes a user query into a structured JSON plan
using an LLM (Qwen / OpenAI-compatible).
"""

from __future__ import annotations #for current versions of python it is not needed but it is good for backward compatibility

import json
from typing import Any #to use any operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from utils.logger import get_logger
from utils.parser import parse_planner_output, PlannerOutput

log = get_logger(__name__)

# ── System Prompt ─────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a Task Planner AI.  Your job is to take a user's request and decompose
it into a structured plan of subtasks.

## Available Tools
{tools_description}

## Instructions
1. Analyse the user's request carefully.
2. Break it into the smallest meaningful subtasks.
3. For each subtask choose the BEST tool from the list above.
   - If no tool fits, use "no_tool" and provide a direct answer in the description.
4. Return ONLY valid JSON — no commentary, no markdown fences.

## Output Format
{{
  "tasks": [
    {{
      "id": 1,
      "description": "Short description of what to do",
      "tool": "tool_name",
      "tool_input": {{ ... }}
    }}
  ]
}}

## Context (previous conversation)
{context}
"""


class PlannerAgent:
    """Uses an LLM to produce a PlannerOutput from a user query."""

    def __init__(self, llm: ChatOpenAI, tools_metadata: list[dict[str, Any]]) -> None:
        self._llm = llm
        self._tools_meta = tools_metadata

    def _build_tools_description(self) -> str:
        lines: list[str] = []
        for t in self._tools_meta:
            lines.append(
                f"- **{t['name']}**: {t['description']}\n"
                f"  Input: {json.dumps(t.get('input_schema', {}), indent=2)}"
            )
        return "\n".join(lines)

    async def plan(self, query: str, context: str = "") -> PlannerOutput:
        """Generate a plan for the given *query*."""
        tools_desc = self._build_tools_description()
        system = _SYSTEM_PROMPT.format(
            tools_description=tools_desc,
            context=context or "(no prior context)",
        )

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=query),
        ]

        log.info("Planner invoked for: %s", query[:120])
        response = await self._llm.ainvoke(messages)
        raw = response.content
        log.debug("Planner raw response:\n%s", raw)

        plan = parse_planner_output(raw)
        return plan
