# backend/services/scraper.py
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import re
import asyncio
from typing import List
from urllib.parse import unquote, urlparse
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
import httpx

# Domains to skip when extracting search result URLs
_SKIP_DOMAINS = {
    "google.com", "google.co.in", "gstatic.com", "googleapis.com",
    "youtube.com", "accounts.google.com", "maps.google.com",
    "support.google.com", "policies.google.com",
}

# Maximum content length per scraped page (characters)
_MAX_CONTENT_LENGTH = 8000


def _extract_markdown(result) -> str:
    """
    Extract the cleanest available markdown from a Crawl4AI result object.

    Handles API changes across Crawl4AI versions by trying multiple
    attribute paths in priority order:
      1. result.markdown_v2.fit_markdown  (legacy ≤0.4)
      2. result.markdown.fit_markdown     (≥0.5 restructured)
      3. result.fit_markdown              (flat attribute)
      4. result.markdown                  (raw fallback, always present)
    """
    # Path 1: legacy markdown_v2 container
    md_v2 = getattr(result, "markdown_v2", None)
    if md_v2:
        fit = getattr(md_v2, "fit_markdown", None)
        if fit and isinstance(fit, str) and fit.strip():
            return fit.strip()

    # Path 2: restructured markdown container (≥0.5)
    md_obj = getattr(result, "markdown", None)
    if md_obj and not isinstance(md_obj, str):
        fit = getattr(md_obj, "fit_markdown", None)
        if fit and isinstance(fit, str) and fit.strip():
            return fit.strip()

    # Path 3: flat attribute
    fit_flat = getattr(result, "fit_markdown", None)
    if fit_flat and isinstance(fit_flat, str) and fit_flat.strip():
        return fit_flat.strip()

    # Path 4: raw markdown string
    if md_obj and isinstance(md_obj, str) and md_obj.strip():
        return md_obj.strip()

    # Final fallback: raw_markdown or empty string
    return getattr(result, "raw_markdown", "") or ""


class NexusScraperService:
    def __init__(self):
        # Prevent headless browser crashes within sandboxed container runtimes
        # text_mode=True and avoid_css=True optimize speed by disabling images and CSS download
        self.browser_config = BrowserConfig(
            headless=True,
            text_mode=True,
            avoid_css=True,
            extra_args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--blink-settings=imagesEnabled=false",
            ]
        )
        # Pruning filter to strip boilerplate, layouts, and tracking scripts
        self.prune_filter = PruningContentFilter(
            threshold=0.45,
            threshold_type="dynamic",
            min_word_threshold=10
        )
        self.markdown_gen = DefaultMarkdownGenerator(content_filter=self.prune_filter)

        self.run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            excluded_tags=["header", "footer", "nav", "aside", "form"],
            word_count_threshold=10,
            exclude_external_links=True,
            markdown_generator=self.markdown_gen,
        )

    # ── Primary: DuckDuckGo search ──────────────────────────────
    def _ddg_search(self, query: str, max_results: int) -> List[str]:
        """Try DuckDuckGo search."""
        # Try default/auto backend first (best for newer ddgs versions)
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                results_list = list(results) if results else []
                urls = [r["href"] for r in results_list if isinstance(r, dict) and "href" in r]
                if urls:
                    print(f"    [INFO] DDG search success using default backend: found {len(urls)} URLs")
                    return urls
        except Exception as e:
            print(f"    [WARN] DDG search with default backend failed: {e}")

        # Fallback to older backend specifications for compatibility
        backends = ["lite", "html", "api"]
        for backend in backends:
            try:
                with DDGS() as ddgs:
                    results = ddgs.text(query, max_results=max_results, backend=backend)
                    results_list = list(results) if results else []
                    urls = [r["href"] for r in results_list if isinstance(r, dict) and "href" in r]
                    if urls:
                        print(f"    [INFO] DDG search success using backend '{backend}': found {len(urls)} URLs")
                        return urls
            except Exception as e:
                print(f"    [WARN] DDG search with backend '{backend}' failed: {e}")
        return []

    # ── Fallback: Google search via httpx ───────────────────────
    def _google_search_fallback(self, query: str, max_results: int) -> List[str]:
        """Scrape Google search results page as a fallback when DDG fails."""
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
            resp = httpx.get(
                "https://www.google.com/search",
                params={"q": query, "num": max_results + 2},
                headers=headers,
                timeout=10.0,
                follow_redirects=True,
            )
            resp.raise_for_status()

            # Extract URLs from Google's /url?q= redirect links
            raw_urls = re.findall(r'/url\?q=([^&"]+)', resp.text)
            urls: List[str] = []
            seen: set[str] = set()
            for raw in raw_urls:
                url = unquote(raw)
                domain = urlparse(url).netloc.lower().replace("www.", "")
                if domain in _SKIP_DOMAINS or domain in seen:
                    continue
                if url.startswith("http"):
                    seen.add(domain)
                    urls.append(url)
                if len(urls) >= max_results:
                    break

            if urls:
                print(f"    [INFO] Google fallback search success: found {len(urls)} URLs")
            return urls
        except Exception as e:
            print(f"    [WARN] Google fallback search failed: {e}")
            return []

    def search_urls(self, query: str, max_results: int = 3) -> List[str]:
        """Search for URLs: tries DDG first, falls back to Google scraping."""
        urls = self._ddg_search(query, max_results)
        if urls:
            return urls

        print("    [INFO] DDG returned no results — trying Google fallback...")
        return self._google_search_fallback(query, max_results)

    async def scrape_url(self, url: str) -> str:
        """Launches an async browser context to extract clean Markdown text."""
        # Try fast HTTP fetch first
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                )
            }
            async with httpx.AsyncClient(headers=headers, timeout=4.0, follow_redirects=True) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    text = resp.text
                    text = re.sub(r'<(script|style).*?>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r'<[^>]*>', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    if len(text) > 100:
                        return text[:_MAX_CONTENT_LENGTH]
        except Exception as e:
            print(f"    [WARN] Fast single HTTP scrape failed: {e}")

        # Fallback to Crawl4AI browser
        try:
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                result = await crawler.arun(url=url, config=self.run_config)
                if result.success:
                    content = _extract_markdown(result)
                    return content[:_MAX_CONTENT_LENGTH] if content else ""
                return f"Scrape Error: {result.error_message}"
        except Exception as e:
            return f"Scrape Exception: {str(e)}"

    async def scrape_urls(self, urls: List[str]) -> List[dict]:
        """Extract clean Markdown text from multiple URLs in parallel."""
        # Try fast HTTP GET requests first
        scraped: List[dict] = []
        async def fetch_one(url: str):
            try:
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/131.0.0.0 Safari/537.36"
                    )
                }
                async with httpx.AsyncClient(headers=headers, timeout=4.0, follow_redirects=True) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        text = resp.text
                        text = re.sub(r'<(script|style).*?>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
                        text = re.sub(r'<[^>]*>', ' ', text)
                        text = re.sub(r'\s+', ' ', text).strip()
                        if len(text) > 100:
                            return {"url": url, "content": text[:_MAX_CONTENT_LENGTH]}
            except Exception as e:
                print(f"    [WARN] Fast HTTP scrape failed for {url}: {e}")
            return None

        try:
            tasks = [fetch_one(url) for url in urls]
            res = await asyncio.gather(*tasks)
            scraped = [r for r in res if r is not None]
        except Exception as e:
            print(f"    [WARN] Error in fast parallel HTTP scrape: {e}")

        if scraped:
            print(f"    [INFO] Fast HTTP scrape successfully fetched {len(scraped)} page(s)")
            return scraped

        print("    [INFO] Fast HTTP scrape failed or returned empty — falling back to Crawl4AI AsyncWebCrawler...")
        # Fallback to Crawl4AI browser
        try:
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                results = await crawler.arun_many(urls=urls, config=self.run_config)
                for result in results:
                    if result.success:
                        content = _extract_markdown(result)
                        if content:
                            scraped.append({
                                "url": result.url,
                                "content": content[:_MAX_CONTENT_LENGTH],
                            })
                    else:
                        print(f"    [WARN] Scraping {result.url} failed: {result.error_message}")
                return scraped
        except Exception as e:
            print(f"    [WARN] Exception during arun_many fallback: {e}")
            return []


nexus_scraper = NexusScraperService()
