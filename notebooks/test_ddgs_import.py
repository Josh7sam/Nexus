from ddgs import DDGS

def test_jetson():
    with DDGS() as ddgs:
        try:
            results = list(ddgs.text("Nvidia Jetson", max_results=5))
            print("ddgs Results:", results)
        except Exception as e:
            print("ddgs Failed:", e)

if __name__ == "__main__":
    test_jetson()
