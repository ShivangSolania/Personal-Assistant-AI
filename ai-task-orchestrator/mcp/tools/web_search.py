"""
mcp/tools/web_search.py
───────────────────────
Simulated web-search tool (MCP-compatible).

In production, replace the handler body with a call to a real search API
(SerpAPI, Brave Search, Tavily, etc.).
"""

from __future__ import annotations

from typing import Any


TOOL_NAME = "web_search"
TOOL_DESCRIPTION = (
    "Search the web for information on a given query. "
    "Returns a list of search results with titles, URLs, and snippets."
)
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query"},
    },
    "required": ["query"],
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "snippet": {"type": "string"},
                },
            },
        },
    },
}


async def handler(query: str, **_kwargs: Any) -> dict[str, Any]:
    """
    Simulated web search.  Replace this body with a real API call in
    production (e.g. httpx.AsyncClient → SerpAPI).
    """
    # Simulated results for demonstration
    simulated_results = [
        {
            "title": f"Result 1 for: {query}",
            "url": f"https://example.com/search?q={query.replace(' ', '+')}",
            "snippet": f"This is a simulated search result providing information about '{query}'. "
                       "In production, this would be a real web search result.",
        },
        {
            "title": f"Result 2 for: {query}",
            "url": f"https://wikipedia.org/wiki/{query.replace(' ', '_')}",
            "snippet": f"Wikipedia article about '{query}' with comprehensive details and references.",
        },
        {
            "title": f"Result 3 for: {query}",
            "url": f"https://docs.example.com/{query.replace(' ', '-')}",
            "snippet": f"Technical documentation and resources related to '{query}'.",
        },
    ]
    return {"results": simulated_results}


def register(registry) -> None:
    """Auto-discovery hook called by ToolRegistry.auto_discover()."""
    registry.register(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=INPUT_SCHEMA,
        output_schema=OUTPUT_SCHEMA,
        handler=handler,
    )
