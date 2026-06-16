import httpx

def test_query(question):
    print(f"\nQuerying: '{question}'...")
    try:
        resp = httpx.post("http://127.0.0.1:8000/chat", json={"question": question}, timeout=45.0)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Answer Preview: {data['answer'][:200]}...")
            print("Sources:", [s['source'] for s in data['sources']])
            print("Metadata:", data['metadata'])
        else:
            print(f"Error Content: {resp.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_query("JP Morgan Chase & CO, what do they do?")
