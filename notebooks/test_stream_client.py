import httpx
import json

def test_stream(question):
    print(f"\nStreaming query: '{question}'...")
    try:
        with httpx.stream("POST", "http://127.0.0.1:8000/chat/stream", json={"question": question}, timeout=45.0) as resp:
            print(f"Status: {resp.status_code}")
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str:
                        event = json.loads(data_str)
                        print(f"EVENT: {event.get('type')} | keys={list(event.keys())}")
                        if event.get("type") == "error":
                            print(f"  -> Error details: {event.get('content')}")
                        elif event.get("type") == "token":
                            # print(event.get("content"), end="", flush=True)
                            pass
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_stream("JP Morgan Chase & CO, what do they do?")
