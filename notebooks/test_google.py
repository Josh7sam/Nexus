import httpx
import re
from urllib.parse import unquote, urlparse

def test_google():
    query = "Nvidia Jetson"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = httpx.get(
        "https://www.google.com/search",
        params={"q": query, "num": 5},
        headers=headers,
        timeout=10.0,
        follow_redirects=True,
    )
    print("Status code:", resp.status_code)
    print("Length of response:", len(resp.text))
    
    # Let's save a bit of response text or look for hrefs
    hrefs = re.findall(r'href="([^"]+)"', resp.text)
    print("Total hrefs found:", len(hrefs))
    
    # Print hrefs that look like results
    http_hrefs = [h for h in hrefs if h.startswith("http") or "/url" in h]
    print("HTTP/url hrefs (first 20):")
    for h in http_hrefs[:20]:
        print(" ", h)
        
    raw_urls = re.findall(r'/url\?q=([^&"]+)', resp.text)
    print("raw_urls found with /url?q=:", len(raw_urls))

if __name__ == "__main__":
    test_google()
