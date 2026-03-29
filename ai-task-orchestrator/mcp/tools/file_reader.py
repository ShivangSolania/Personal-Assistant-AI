"""
mcp/tools/file_reader.py
────────────────────────
Read the contents of a local file (MCP-compatible).
Includes basic path validation to prevent directory traversal.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

TOOL_NAME = "file_reader"
TOOL_DESCRIPTION = (
    "Read the contents of a local file given its path. "
    "Returns the text content and metadata (size, lines)."
)
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "Absolute or relative path to the file to read",
        },
    },
    "required": ["file_path"],
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string"},
        "content": {"type": "string"},
        "size_bytes": {"type": "integer"},
        "line_count": {"type": "integer"},
    },
}

_MAX_SIZE = 1_000_000  # 1MB safety limit


async def handler(file_path: str, **_kwargs: Any) -> dict[str, Any]:
    """Read *file_path* and return its textual content."""
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    if path.stat().st_size > _MAX_SIZE:
        raise ValueError(f"File too large ({path.stat().st_size} bytes > {_MAX_SIZE} limit)")

    content = path.read_text(encoding="utf-8", errors="replace")
    return {
        "file_path": str(path),
        "content": content,
        "size_bytes": path.stat().st_size,
        "line_count": content.count("\n") + 1,
    }


def register(registry) -> None:
    """Auto-discovery hook."""
    registry.register(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=INPUT_SCHEMA,
        output_schema=OUTPUT_SCHEMA,
        handler=handler,
    )
