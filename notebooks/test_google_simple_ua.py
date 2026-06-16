import httpx
from bs4 import BeautifulSoup
import re

def try_ua(ua):
    headers = {"User-Agent": ua} if ua else {}
    try:
        resp = httpx.get("https://www.google.com/search", params={"q": "Nvidia Jetson", "num": 5}, headers=headers, timeout=10.0)
        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string if soup.title else "No title"
        
        # Check /url?q= pattern
        raw_urls = re.findall(r'/url\?q=([^&"]+)', resp.text)
        print(f"UA: {ua[:40]:<40} | Status: {resp.status_code} | Length: {len(resp.text):<6} | Title: {title:<25} | /url?q= count: {len(raw_urls)}")
    except Exception as e:
        print(f"UA: {ua[:40]:<40} | Failed: {e}")

if __name__ == "__main__":
    try_ua("python-httpx/0.28.1")
    try_ua("curl/8.4.0")
    try_ua("Wget/1.21.1")
    try_ua("Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.1")
    try_ua("Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)")
    try_ua("Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36")
