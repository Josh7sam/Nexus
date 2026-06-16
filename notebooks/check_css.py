import re

with open(r"c:\Users\HP\OneDrive\Documents\NEXUS - Hybrid Agentic Retrieval System\frontend\style.css", "r", encoding="utf-8") as f:
    content = f.read()

# Strip comments to avoid false positives inside comments
content_clean = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

stack = []
lines = content.split('\n')
current_line = 0

for char_idx, char in enumerate(content):
    if char == '\n':
        current_line += 1
    if char == '{':
        stack.append(current_line)
    elif char == '}':
        if not stack:
            print(f"Extra closing brace '}}' at line {current_line + 1}")
        else:
            stack.pop()

if stack:
    print(f"Unmatched opening braces '{{' from lines: {[l + 1 for l in stack]}")
else:
    print("All braces match perfectly!")
