import sys
import os
import asyncio

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

# Monkey patch or test import
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

from services.scraper import NexusScraperService

async def main():
    scraper = NexusScraperService()
    # Override ddg search to use the new DDGS import and default backend
    def custom_ddg_search(query: str, max_results: int):
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                results_list = list(results) if results else []
                urls = [r["href"] for r in results_list if isinstance(r, dict) and "href" in r]
                return urls
        except Exception as e:
            print("Custom DDG failed:", e)
            return []
            
    query = "Nvidia Jetson"
    print(f"Searching for query: {query}")
    urls = custom_ddg_search(query, 2)
    print("Found URLs:", urls)
    
    if urls:
        print("Scraping URLs with crawl4ai...")
        results = await scraper.scrape_urls(urls)
        print(f"Scraped {len(results)} pages:")
        for r in results:
            print(f"URL: {r['url']}")
            print(f"Content length: {len(r['content'])}")
            print(f"Preview: {r['content'][:300]}")
            print("="*40)
    else:
        print("No URLs to scrape!")

if __name__ == "__main__":
    asyncio.run(main())
