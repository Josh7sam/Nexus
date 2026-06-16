import httpx
import json

url = "http://localhost:8000/chat"
payload = {
    "question": "What is Retrieval-Augmented Generation?"
}

print("Sending request to /chat...")
try:
    response = httpx.post(url, json=payload, timeout=60.0)
    print("Status Code:", response.status_code)
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print("Request failed:", e)
