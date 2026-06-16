import httpx
import re
from urllib.parse import unquote, urlparse

def try_google(user_agent=None):
    headers = {}
    if user_agent:
        headers["User-Agent"] = user_agent
    headers["Accept-Language"] = "en-US,en;q=0.9"
    
    try:
        resp = httpx.get(
            "https://www.google.com/search",
            params={"q": "Nvidia Jetson", "num": 5},
            headers=headers,
            timeout=10.0,
            follow_redirects=True,
        )
        print(f"\nUser-Agent: {user_agent or 'None'}")
        print("Status:", resp.status_code)
        
        # Check title
        title = re.search(r'<title>(.*?)</title>', resp.text)
        print("Title:", title.group(1) if title else "None")
        
        # Check for /url?q= pattern
        raw_urls = re.findall(r'/url\?q=([^&"]+)', resp.text)
        print("Urls found with /url?q=:", len(raw_urls))
        
        # Check for standard hrefs
        hrefs = re.findall(r'href="([^"]+)"', resp.text)
        http_hrefs = [h for h in hrefs if h.startswith("http") or "/url" in h]
        print("Total http/url hrefs:", len(http_hrefs))
        if http_hrefs:
            print("First 3 hrefs:")
            for h in http_hrefs[:3]:
                print(" ", h[:100])
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    # Test a few different User-Agents
    try_google(None)
    try_google("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    try_google("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0")
    try_google("Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)")
    try_google("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15")
