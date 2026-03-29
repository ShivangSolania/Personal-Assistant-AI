"""
mcp/registry.py
───────────────
MCP Tool Registry — registers, discovers, and looks up tools by name.

Every tool is a dict conforming to:
    {
        "name":          str,
        "description":   str,
        "input_schema":  dict   (JSON Schema),
        "output_schema": dict   (JSON Schema),
        "handler":       Callable[..., Awaitable[dict]],
    }
"""

from __future__ import annotations

import importlib
import os
import pkgutil #for auto-discovering tools
from typing import Any, Callable, Awaitable

from utils.logger import get_logger

log = get_logger(__name__)


class ToolRegistry:
    """Central registry for all MCP-compatible tools."""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    # ── Registration ──────────────────────────────────────

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict,
        output_schema: dict,
        handler: Callable[..., Awaitable[dict]],
    ) -> None:
        """Register a single tool."""
        self._tools[name] = {
            "name": name,
            "description": description,
            "input_schema": input_schema,
            "output_schema": output_schema,
            "handler": handler,
        }
        log.info("Registered tool: %s", name)

    # ── Lookup ────────────────────────────────────────────

    def get(self, name: str) -> dict[str, Any] | None:
        """Return the tool dict for *name*, or None."""
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        """Return metadata (no handler) for every registered tool."""
        return [
            {k: v for k, v in t.items() if k != "handler"}
            for t in self._tools.values()
        ]

    def tool_names(self) -> list[str]:
        """Return a list of registered tool names."""
        return list(self._tools.keys())

    # ── Auto-Discovery ────────────────────────────────────

    def auto_discover(self, package_path: str | None = None) -> None:
        """
        Walk the ``mcp.tools`` package and call every module's
        ``register(registry)`` function so tools self-register.
        """
        if package_path is None:
            package_path = os.path.join(os.path.dirname(__file__), "tools")

        tools_package = "mcp.tools"
        log.info("Auto-discovering tools in %s …", package_path)

        for importer, module_name, is_pkg in pkgutil.iter_modules([package_path]):
            if module_name.startswith("_"):
                continue
            full_name = f"{tools_package}.{module_name}"
            try:
                mod = importlib.import_module(full_name)
                if hasattr(mod, "register"):
                    mod.register(self)        # type: ignore[attr-defined]
                    log.info("Auto-registered tool module: %s", full_name)
                else:
                    log.warning("Module %s has no register() — skipped", full_name)
            except Exception as exc:
                log.error("Failed to import %s: %s", full_name, exc)
