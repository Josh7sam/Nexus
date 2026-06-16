import sys
import os
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from duckduckgo_search import DDGS

def test_direct():
    query = "definition and scientific explanation of a nebula in astronomy and space science"
    backends = ["lite", "html", "api"]
    for b in backends:
        print(f"\n--- Testing backend: {b} ---")
        try:
            with DDGS() as ddgs:
                gen = ddgs.text(query, max_results=2, backend=b)
                print("Type of gen:", type(gen))
                results = list(gen)
                print(f"Results: {results}")
        except Exception as e:
            print(f"Failed with exception: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    test_direct()
