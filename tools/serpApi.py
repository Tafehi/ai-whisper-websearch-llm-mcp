# serp_mcp.py
from __future__ import annotations

import os
from typing import Any, Dict, List

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables from .env if present
load_dotenv()

API_KEY = os.getenv("SERP_SEARCH_API")
BASE_URL = os.getenv("SERP_SEARCH_URL")
TIMEOUT = float(os.getenv("SERP_SEARCH_TIMEOUT", "10"))

# Initialize MCP tool server
mcp = FastMCP(name="serpSearch", host="localhost", port=8001)


def _extract_results(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract title/link/snippet triples from typical SerpAPI-shaped JSON."""
    if not isinstance(data, dict):
        return []

    if "error" in data and isinstance(data["error"], str):
        return [{"title": "", "link": "", "snippet": data["error"]}]

    results = []
    for item in data.get("organic_results", []):
        title = item.get("title", "") or item.get("heading", "")
        link = item.get("link") or item.get("url") or ""
        snippet = item.get("snippet") or item.get("description") or ""
        if title or link or snippet:
            results.append({"title": title, "link": link, "snippet": snippet})
    return results


@mcp.tool()
async def search_serpapi(query: str) -> Dict[str, Any]:
    """
    Search the web using the configured SerpAPI-compatible endpoint.

    Args:
        query: The search query string.

    Returns:
        JSON with either:
          - {"results": [{title, link, snippet}, ...], "count": <int>}
          - {"error": <string>, ...}
    """
    if not BASE_URL or not API_KEY:
        return {
            "error": "Missing SERPSEARCH_API or SERPSEARCH_URL environment variables."
        }

    params = {"q": query, "api_key": API_KEY}

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            results = _extract_results(data)
            return {"results": results, "count": len(results)}
    except httpx.ReadTimeout:
        return {"error": "Request to SerpAPI timed out."}
    except httpx.HTTPStatusError as e:
        return {
            "error": "HTTP error from search backend.",
            "status_code": e.response.status_code,
            "details": e.response.text,
        }
    except Exception as e:
        return {"error": f"Unexpected error: {e.__class__.__name__}: {e}"}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
