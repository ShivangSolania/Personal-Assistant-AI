"""
mcp/tools/custom_api.py
───────────────────────
Generic HTTP API caller (MCP-compatible).
Uses httpx for async requests to any REST endpoint.
"""

from __future__ import annotations

from typing import Any

import httpx

TOOL_NAME = "custom_api"
TOOL_DESCRIPTION = (
    "Make an HTTP request to an external API. "
    "Supports GET and POST methods with optional headers and JSON body."
)
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "The API endpoint URL"},
        "method": {
            "type": "string",
            "enum": ["GET", "POST"],
            "description": "HTTP method (default: GET)",
        },
        "headers": {
            "type": "object",
            "description": "Optional HTTP headers",
            "additionalProperties": {"type": "string"},
        },
        "body": {
            "type": "object",
            "description": "Optional JSON body (for POST requests)",
        },
    },
    "required": ["url"],
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status_code": {"type": "integer"},
        "headers": {"type": "object"},
        "body": {"type": "string"},
    },
}

_TIMEOUT = 30  # seconds


async def handler(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Execute an HTTP request and return the response."""
    method = method.upper()
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        if method == "POST":
            resp = await client.post(url, headers=headers, json=body)
        else:
            resp = await client.get(url, headers=headers)

    # Truncate very large responses
    response_body = resp.text[:10_000] if len(resp.text) > 10_000 else resp.text

    return {
        "status_code": resp.status_code,
        "headers": dict(resp.headers),
        "body": response_body,
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
