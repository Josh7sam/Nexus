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

from duckduckgo_search import DDGS

async def test_ddg():
    print("Testing DDGS API directly...")
    try:
        with DDGS() as ddgs:
            results = ddgs.text("Gemini AI", max_results=3)
            # Try converting to list or iterating
            results_list = list(results)
            print(f"Results type: {type(results)}")
            print(f"Found {len(results_list)} results.")
            for r in results_list:
                print(f" - href: {r.get('href')}")
    except Exception as e:
        print(f"DDG Search failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ddg())
