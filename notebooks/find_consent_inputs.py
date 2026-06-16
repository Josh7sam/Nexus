from bs4 import BeautifulSoup

with open("scratch/google_response.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")
print("Forms found:", len(soup.find_all("form")))
for i, form in enumerate(soup.find_all("form")):
    print(f"Form {i}: action='{form.get('action')}', method='{form.get('method')}'")
    for inp in form.find_all("input"):
        print(f"  Input: name='{inp.get('name')}', type='{inp.get('type')}', value='{inp.get('value')}'")

print("\nButtons:")
for b in soup.find_all("button"):
    print(f"  Button: text='{b.get_text().strip()}', name='{b.get('name')}', type='{b.get('type')}'")
    
# Let's write the first 5000 characters of the body to another file so we can inspect
with open("scratch/body_snippet.txt", "w", encoding="utf-8") as f:
    f.write(soup.body.prettify()[:10000] if soup.body else "No body")
print("Saved body snippet.")
