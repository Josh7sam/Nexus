import httpx
from bs4 import BeautifulSoup
import re
from urllib.parse import unquote

def print_msie():
    headers = {"User-Agent": "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)"}
    resp = httpx.get("https://www.google.com/search", params={"q": "Nvidia Jetson", "num": 5}, headers=headers, timeout=10.0)
    print("MSIE 6.0 response length:", len(resp.text))
    
    soup = BeautifulSoup(resp.text, "html.parser")
    print("Links:")
    for a in soup.find_all("a"):
        href = a.get("href", "")
        text = a.get_text().strip()
        print(f" - {text}: {href}")
        
    print("\nBody text:")
    print(soup.body.get_text() if soup.body else "No body")

if __name__ == "__main__":
    print_msie()
