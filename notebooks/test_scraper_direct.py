import sys
import os
import asyncio

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from services.scraper import nexus_scraper

async def test_scraper():
    query = "Google DeepMind Gemini news"
    print(f"1. Searching URLs for query: '{query}'...")
    urls = nexus_scraper.search_urls(query, max_results=2)
    print(f"Found URLs: {urls}")
    
    if not urls:
        print("No URLs found from search! Let's try searching using DDGS directly with fallback formats.")
        from duckduckgo_search import DDGS
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                print(f"Direct DDGS results: {results}")
        except Exception as e:
            print(f"Direct DDGS failure: {e}")
        return

    print("\n2. Scraping URLs...")
    try:
        results = await nexus_scraper.scrape_urls(urls)
        print(f"Scrape results count: {len(results)}")
        for idx, res in enumerate(results):
            print(f"\nResult {idx} URL: {res['url']}")
            print(f"Content length: {len(res['content'])}")
            print(f"Preview: {res['content'][:200]}...")
    except Exception as e:
        print(f"Scrape execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_scraper())
