from duckduckgo_search import DDGS

def test_jetson():
    with DDGS() as ddgs:
        try:
            results = ddgs.text("Nvidia Jetson", max_results=5)
            print("Without backend parameter:")
            print("Results:", results)
        except Exception as e:
            print("Failed without backend:", e)
            
        for b in ["lite", "html", "api", "auto"]:
            try:
                results = ddgs.text("Nvidia Jetson", max_results=5, backend=b)
                print(f"Backend '{b}' results count: {len(list(results))}")
            except Exception as e:
                print(f"Backend '{b}' failed: {e}")

if __name__ == "__main__":
    test_jetson()
