import httpx
import re
from urllib.parse import unquote, urlparse

def test_google():
    query = "Nvidia Jetson"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)",
        "Accept-Language": "en-US,en;q=0.9",
    }
    url = "https://www.google.com/search"
    params = {"q": query, "num": 5}
    try:
        resp = httpx.get(url, params=params, headers=headers, timeout=10.0)
        print("Google Status:", resp.status_code)
        raw_urls = re.findall(r'/url\?q=([^&"]+)', resp.text)
        print("Found raw urls count:", len(raw_urls))
        urls = []
        for raw in raw_urls:
            url_decoded = unquote(raw)
            if url_decoded.startswith("http") and not any(d in url_decoded for d in ["google.com", "google.co.in", "gstatic.com"]):
                urls.append(url_decoded)
        print("Filtered URLs:", urls)
    except Exception as e:
        print("Google search failed:", e)

if __name__ == "__main__":
    test_google()
