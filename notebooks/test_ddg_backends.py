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

async def test_backends():
    query = "Gemini AI"
    backends = ["api", "html", "lite"]
    
    for b in backends:
        print(f"\n--- Testing backend: '{b}' ---")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3, backend=b))
                print(f"[{b}] Found {len(results)} results:")
                for r in results:
                    print(f" - {r.get('href')}")
        except Exception as e:
            print(f"[{b}] Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_backends())
