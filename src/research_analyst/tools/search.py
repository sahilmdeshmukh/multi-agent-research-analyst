from __future__ import annotations

import os
import time

from tavily import TavilyClient


def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web via Tavily and return a list of result dicts.

    Retries once on rate-limit (HTTP 429) with a 60-second back-off.
    Each result dict has keys: url, title, content.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")

    client = TavilyClient(api_key=api_key)

    for attempt in range(2):
        try:
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=False,
            )
            results = response.get("results", [])
            return [
                {
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                }
                for r in results
            ]
        except Exception as exc:
            if attempt == 0 and "429" in str(exc):
                time.sleep(60)
                continue
            raise
    return []
