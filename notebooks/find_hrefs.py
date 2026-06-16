import re
from bs4 import BeautifulSoup

with open("scratch/google_response.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")
print("Page title:", soup.title.string if soup.title else "No title")

# Find all links
all_a = soup.find_all("a")
print("Total <a> tags:", len(all_a))

# Print first 20 <a> tags and their hrefs
for i, a in enumerate(all_a[:40]):
    href = a.get("href", "")
    text = a.get_text().strip()
    print(f"{i}: text='{text}', href='{href}'")
