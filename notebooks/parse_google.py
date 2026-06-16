import re

with open("scratch/google_response.html", "r", encoding="utf-8") as f:
    html = f.read()

print("HTML length:", len(html))

# Let's see if there are standard links starting with http or https in href attribute
links = re.findall(r'href="([^"]+)"', html)
print(f"Total hrefs: {len(links)}")

# Print links that have "google" in them or not:
google_links = [l for l in links if "google" in l]
non_google_links = [l for l in links if "google" not in l]

print(f"Google links count: {len(google_links)}")
print(f"Non-google links count: {len(non_google_links)}")

print("\nNon-Google links first 10:")
for l in non_google_links[:10]:
    print(" ", l)

print("\nAll links first 20:")
for l in links[:20]:
    print(" ", l[:100])
