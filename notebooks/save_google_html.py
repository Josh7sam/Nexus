import httpx

def save_google_html():
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
    with open("scratch/google_response.html", "w", encoding="utf-8") as f:
        f.write(resp.text)
    print("Saved HTML of length:", len(resp.text))
    
    # print the first 500 characters and look for <title>
    import re
    title = re.search(r'<title>(.*?)</title>', resp.text)
    if title:
        print("Page Title:", title.group(1))
    else:
        print("No title found")

if __name__ == "__main__":
    save_google_html()
