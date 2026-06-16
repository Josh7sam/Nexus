from bs4 import BeautifulSoup

with open("scratch/google_response.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")
# get text
text = soup.get_text()
# print non-empty lines
lines = [l.strip() for l in text.splitlines() if l.strip()]
print("First 30 lines of text in the page:")
for l in lines[:30]:
    print(" ", l)
