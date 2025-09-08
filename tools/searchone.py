# searchone_server.py (example path)
import os
import httpx
from typing import List, Optional
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Initialize MCP tool server
mcp = FastMCP(name="searchone", host="localhost", port=8001)

class SearchOneApiSearch:
    def __init__(self):
        load_dotenv()
        # API key is optional for keyless tier
        self._API_KEY = os.getenv("SEARCH1_API_KEY")
        self._BASE_URL = os.getenv("SEARCH1_API_URL")

    async def search_one_impl(
        self,
        query: str,
        search_service: str = "google",
        max_results: int = 5,
        crawl_results: int = 0,
        image: bool = False,
        include_sites: Optional[str] = "",
        exclude_sites: Optional[str] = "",
        language: str = "en",
        time_range: Optional[str] = None
    ) -> str:

        """
        Search the web using SearchOne (search1api).

        Args:
            query: Search query (required)
            search_service: 'google' | 'bing' | 'duckduckgo' (default: google)
            max_results: Number of links to return (default: 5)
            crawl_results: Number of links to crawl for content (default: 0)
            image: Set True to search images (default: False)
            include_sites: Comma-separated allowlist of domains
            exclude_sites: Comma-separated blocklist of domains
            language: ISO language code (default: 'en')
            time_range: e.g., 'day' | 'week' | 'month' | 'year' (optional)
        """

        if not self._BASE_URL:
            return "Missing SEARCH_ONE_URL in environment variables."

        headers = {"Content-Type": "application/json"}
        if self._API_KEY:
            headers["Authorization"] = f"Bearer {self._API_KEY}"

        payload = {
            "query": query,
            "search_service": search_service,
            "max_results": max_results,
            "crawl_results": crawl_results,
            "image": image,
            "language": language
        }

        # Convert comma-separated strings to lists if provided
        if include_sites:
            payload["include_sites"] = [s.strip() for s in include_sites.split(",") if s.strip()]
        if exclude_sites:
            payload["exclude_sites"] = [s.strip() for s in exclude_sites.split(",") if s.strip()]
        if time_range:
            payload["time_range"] = time_range

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(self._BASE_URL, headers=headers, json=payload)
                resp.raise_for_status()
                try:
                    data = resp.json()
                except ValueError:
                    return f"Non-JSON response from SearchOne: {resp.text[:500]}"

            snippets = self.extract_snippets_searchone(data)
            return "\n\n".join(snippets) if snippets else "No relevant results found."

        except httpx.TimeoutException:
            return "Request to SearchOne timed out."
        except httpx.HTTPStatusError as e:
            return f"HTTP error: {e.response.status_code} - {e.response.text[:500]}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    @staticmethod
    def extract_snippets_searchone(data: dict) -> List[str]:
        """
        Extracts human-friendly lines from SearchOne response.
        Handles multiple possible shapes defensively.
        """

        # Common error container
        if isinstance(data, dict) and "error" in data:
            return [str(data["error"])]

        # Likely containers for results
        candidates = []
        if isinstance(data, dict):
            for k in ("results", "data", "organic_results"):
                if k in data and isinstance(data[k], list):
                    candidates = data[k]
                    break
        elif isinstance(data, list):
            candidates = data

        lines: List[str] = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            title = item.get("title") or item.get("name") or ""
            url = item.get("url") or item.get("link") or item.get("source_url") or ""
            source = item.get("source") or item.get("domain") or ""
            snippet = (
                item.get("snippet")
                or item.get("summary")
                or item.get("content")
                or item.get("description")
                or ""
            )

            header_bits = [b for b in [title, url, source] if b]
            header = " • ".join(header_bits) if header_bits else ""
            if header and snippet:
                lines.append(f"{header}\n{snippet}")
            elif header:
                lines.append(header)
            elif snippet:
                lines.append(snippet)

        # Fallback for crawled content blocks
        if not lines and isinstance(data, dict):
            crawl = data.get("crawled_results") or data.get("crawl_results")
            if isinstance(crawl, list):
                for c in crawl:
                    url = c.get("url") or ""
                    text = c.get("text") or c.get("content") or ""
                    if url or text:
                        lines.append(f"{url}\n{text[:500]}")

        return lines


# Create the backend instance
_search_backend = SearchOneApiSearch()

# Register a TOP-LEVEL tool (no 'self' in signature!)
@mcp.tool(name="search_one", description="Search the web using SearchOne (search1api).")
async def search_one(
    query: str,
    search_service: str = "google",
    max_results: int = 5,
    crawl_results: int = 0,
    image: bool = False,
    include_sites: Optional[str] = "",
    exclude_sites: Optional[str] = "",
    language: str = "en",
    time_range: Optional[str] = None
) -> str:
    return await _search_backend.search_one_impl(
        query=query,
        search_service=search_service,
        max_results=max_results,
        crawl_results=crawl_results,
        image=image,
        include_sites=include_sites,
        exclude_sites=exclude_sites,
        language=language,
        time_range=time_range,
    )


if __name__ == "__main__":
    # Start MCP over streamable HTTP
    mcp.run(transport="streamable-http")
