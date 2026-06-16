import sys
import os
import asyncio

# Force UTF-8 encoding for stdout
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from services.scraper import nexus_scraper

async def test_crawl():
    url = "https://gemini.google/us/about/?hl=en"
    print(f"Scraping URL: {url}...")
    try:
        content = await nexus_scraper.scrape_url(url)
        print(f"Success! Content length: {len(content)}")
        print(f"Preview:\n{content[:500]}...")
    except Exception as e:
        print(f"Failed to scrape: {e}")

if __name__ == "__main__":
    asyncio.run(test_crawl())
