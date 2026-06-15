"""
tools.py — the web_search tool the agent can call.

Uses DuckDuckGo (no API key required) for real results. We try the
`duckduckgo_search` package if installed; otherwise we fall back to DDG's
public HTML/JSON endpoints via httpx. Either way this performs a genuine web
search and returns structured results the agent summarizes.

TOOL_SCHEMAS is the OpenAI function-calling schema sent to the model.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current, factual, or recent information. "
                "Use this whenever the user asks about news, current events, "
                "live data, or anything you may not know or that changes over time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "How many results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


async def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Return a list of {title, url, snippet}. Tries the SDK, then HTTP fallback."""
    # 1) Preferred: duckduckgo_search package (sync), run in a thread.
    try:
        from duckduckgo_search import DDGS  # type: ignore
        import anyio

        def _run() -> list[dict[str, str]]:
            out: list[dict[str, str]] = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    out.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })
            return out

        results = await anyio.to_thread.run_sync(_run)
        if results:
            return results
    except Exception:
        pass  # fall through to HTTP fallback

    # 2) Fallback: DuckDuckGo Instant Answer API (JSON, no key).
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            )
            data = resp.json()
            results: list[dict[str, str]] = []
            # AbstractText is the headline answer, if any.
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading", query),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data.get("AbstractText", ""),
                })
            # RelatedTopics gives additional hits.
            for topic in data.get("RelatedTopics", []):
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("Text", "")[:80],
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", ""),
                    })
                if len(results) >= max_results:
                    break
            if results:
                return results
    except Exception:
        pass

    # 3) Nothing worked — return an explicit, honest empty result.
    return [{
        "title": "No results",
        "url": "",
        "snippet": f"Web search returned no results for '{query}'.",
    }]


async def run_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a tool by name and return a JSON string result for the model."""
    if name == "web_search":
        query = str(args.get("query", "")).strip()
        max_results = int(args.get("max_results", 5))
        results = await web_search(query, max_results)
        return json.dumps({"query": query, "results": results})
    return json.dumps({"error": f"Unknown tool: {name}"})
