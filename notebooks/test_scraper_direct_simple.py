import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from services.scraper import nexus_scraper

def test():
    query = "definition and scientific explanation of a nebula in astronomy and space science"
    print("Querying via nexus_scraper.search_urls:")
    urls = nexus_scraper.search_urls(query, max_results=2)
    print("URLs found:", urls)

if __name__ == "__main__":
    test()
