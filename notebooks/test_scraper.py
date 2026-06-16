import sys
import os
import asyncio

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from services.scraper import nexus_scraper

async def main():
    query = "Nvidia Jetson"
    print("Searching for URLs...")
    urls = nexus_scraper.search_urls(query, max_results=2)
    print("Found URLs:", urls)
    
    if urls:
        print("Scraping URLs...")
        results = await nexus_scraper.scrape_urls(urls)
        print("Scrape results:")
        for r in results:
            print(f"URL: {r['url']}")
            print(f"Content length: {len(r['content'])}")
            print(f"Preview: {r['content'][:200]}...")
            print("-" * 50)
    else:
        print("No URLs found!")

if __name__ == "__main__":
    asyncio.run(main())
