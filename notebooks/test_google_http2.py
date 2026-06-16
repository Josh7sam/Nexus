import httpx
from bs4 import BeautifulSoup
import re

def test_google_http2():
    query = "Nvidia Jetson"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }
    
    try:
        # Use http2=True
        with httpx.Client(http2=True, headers=headers, follow_redirects=True) as client:
            resp = client.get("https://www.google.com/search", params={"q": query, "num": 5})
            print("HTTP2 Status:", resp.status_code)
            print("Length:", len(resp.text))
            
            soup = BeautifulSoup(resp.text, "html.parser")
            print("Title:", soup.title.string if soup.title else "No title")
            
            # Find non-google links
            links = []
            for a in soup.find_all("a"):
                href = a.get("href", "")
                if href.startswith("http") and "google.com" not in href:
                    links.append((a.get_text().strip(), href))
                elif href.startswith("/url?q="):
                    # Extract the actual URL
                    from urllib.parse import unquote, urlparse
                    match = re.search(r'/url\?q=([^&"]+)', href)
                    if match:
                        url = unquote(match.group(1))
                        if "google.com" not in url:
                            links.append((a.get_text().strip(), url))
                            
            print("Found links:", len(links))
            for t, h in links[:10]:
                print(f" - {t}: {h}")
    except Exception as e:
        print("Failed:", e)

if __name__ == "__main__":
    test_google_http2()
