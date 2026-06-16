import httpx
from bs4 import BeautifulSoup
import re

def test_google_follow_href():
    query = "Nvidia Jetson"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    client = httpx.Client(headers=headers, follow_redirects=True)
    
    # Request 1
    resp = client.get("https://www.google.com/search", params={"q": query, "num": 5})
    print("Request 1 Status:", resp.status_code)
    
    soup = BeautifulSoup(resp.text, "html.parser")
    # Let's search for a link that has '/search' in it
    redirect_link = None
    for a in soup.find_all("a"):
        href = a.get("href", "")
        if "/search" in href:
            redirect_link = href
            break
            
    if redirect_link:
        if redirect_link.startswith("/"):
            redirect_url = "https://www.google.com" + redirect_link
        else:
            redirect_url = redirect_link
            
        print("Following redirect link:", redirect_url)
        # Request 2
        resp2 = client.get(redirect_url)
        print("Request 2 Status:", resp2.status_code)
        print("Request 2 Length:", len(resp2.text))
        
        # Parse links in Request 2
        soup2 = BeautifulSoup(resp2.text, "html.parser")
        print("Request 2 Title:", soup2.title.string if soup2.title else "No title")
        
        # Check for search result links
        # Search results typically have <a href="..."> but we should filter out google.com domains
        links = []
        for a in soup2.find_all("a"):
            href = a.get("href", "")
            if href.startswith("http") and "google.com" not in href:
                links.append((a.get_text().strip(), href))
                
        print("Found non-google links in Request 2:", len(links))
        for text, href in links[:10]:
            print(f" - text='{text}', href='{href}'")
            
        # Check /url?q= pattern in Request 2
        raw_urls = re.findall(r'/url\?q=([^&"]+)', resp2.text)
        print("raw_urls with /url?q= in Request 2:", len(raw_urls))
    else:
        print("No redirect link found in Request 1!")

if __name__ == "__main__":
    test_google_follow_href()
