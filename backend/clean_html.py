import os
import re

# Resolve paths relative to this script's location
base_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_dir, "model_configuration.html")
output_path = os.path.join(base_dir, "model_configuration_cleaned.html")

if not os.path.exists(input_path):
    print(f"File not found: {input_path}")
else:
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace long base64 strings
    content_cleaned = re.sub(r'data:image/[a-zA-Z]+;base64,[a-zA-Z0-9+/=]+', '[BASE64_IMAGE]', content)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content_cleaned)

    print("Cleaned file written. Total length reduced from", len(content), "to", len(content_cleaned))
