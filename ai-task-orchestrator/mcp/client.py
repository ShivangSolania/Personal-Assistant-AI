"""
mcp/client.py
─────────────
MCP Client — formats requests, dispatches to the appropriate tool handler,
and returns a normalised response envelope.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)


class MCPClient:
    """
    Bridges between the Executor agent and the Tool Registry.

    Usage::

        client = MCPClient(registry)
        result = await client.call_tool("calculator", {"expression": "2+2"})
    """

    def __init__(self, registry) -> None:          # avoid circular import
        self._registry = registry

    # ── Public API ────────────────────────────────────────

    async def call_tool(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """
        Format an MCP request, dispatch it, and return a response envelope.

        Returns
        -------
        dict with keys: ``tool``, ``status``, ``result`` | ``error``, ``elapsed_ms``
        """
        tool = self._registry.get(tool_name)
        if tool is None:
            available = ", ".join(self._registry.tool_names()) or "(none)"
            error_msg = f"Tool '{tool_name}' not found. Available: {available}"
            log.error(error_msg)
            return self._envelope(tool_name, "error", error=error_msg)

        # ── Build MCP request ─────────────────────────────
        mcp_request = {
            "tool": tool_name,
            "input": tool_input,
        }
        log.info("MCP request  → %s(%s)", tool_name, tool_input)

        # ── Dispatch ──────────────────────────────────────
        start = time.perf_counter()
        try:
            handler = tool["handler"]
            result = await handler(**tool_input)
            elapsed = (time.perf_counter() - start) * 1000
            log.info("MCP response ← %s  (%.1f ms)", tool_name, elapsed)
            return self._envelope(tool_name, "success", result=result, elapsed_ms=elapsed)

        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            log.error("Tool %s failed after %.1f ms: %s", tool_name, elapsed, exc)
            return self._envelope(tool_name, "error", error=str(exc), elapsed_ms=elapsed)

    def list_available_tools(self) -> list[dict[str, Any]]:
        """Return metadata for every registered tool (for the Planner)."""
        return self._registry.list_tools()

    # ── Private ───────────────────────────────────────────

    @staticmethod
    def _envelope(
        tool: str,
        status: str,
        *,
        result: Any = None,
        error: str | None = None,
        elapsed_ms: float = 0,
    ) -> dict[str, Any]:
        env: dict[str, Any] = {"tool": tool, "status": status, "elapsed_ms": round(elapsed_ms, 1)}
        if status == "success":
            env["result"] = result
        else:
            env["error"] = error
        return env
