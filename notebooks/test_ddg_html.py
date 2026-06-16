import httpx
import re
import urllib.parse

def search_ddg_html(query):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    print(f"Requesting: {url}")
    try:
        r = httpx.get(url, headers=headers, timeout=10.0)
        print("Status code:", r.status_code)
        # Find links. In DDG HTML, links are usually of the format:
        # <a class="result__snippet" href="url"> or similar.
        # Let's search for hrefs in the result class container
        # Format in DDG HTML is <a class="result__url" href="URL">
        links = re.findall(r'href="([^"]+)"', r.text)
        print(f"Total hrefs found: {len(links)}")
        
        # Let's filter urls that start with http/https and are not ddg domains
        urls = []
        for link in links:
            # Decode URL if it has ddg redirect like /l/?kh=-1&uddg=HTTPS_URL
            if "uddg=" in link:
                parsed = urllib.parse.urlparse(link)
                qs = urllib.parse.parse_qs(parsed.query)
                if "uddg" in qs:
                    link = qs["uddg"][0]
            
            if link.startswith("http") and not any(d in link for d in ["duckduckgo.com", "ddg.gg"]):
                if link not in urls:
                    urls.append(link)
                    if len(urls) >= 3:
                        break
        print("Extracted URLs:", urls)
        return urls
    except Exception as e:
        print("Failed to search DDG HTML:", e)
        return []

if __name__ == "__main__":
    search_ddg_html("Who is Moses")
