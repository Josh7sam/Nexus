import re

with open("backend/api/server.py", "r", encoding="utf-8") as f:
    code = f.read()

print("=== Matches for sanitize_title in server.py ===")
for m in re.finditer(r"sanitize_title", code):
    start = max(0, m.start() - 100)
    end = min(len(code), m.end() + 150)
    print(code[start:end])
    print("-----------------")
