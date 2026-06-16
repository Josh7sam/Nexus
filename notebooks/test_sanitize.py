# Test script for title sanitization edge cases

FILLER_PREFIXES = [
    "based on the provided data",
    "based on the provided",
    "based on the documents",
    "based on the information",
    "according to the context",
    "according to the documents",
    "the provided data",
    "based on",
]

def sanitize_title(title_str: str,
                   full_response: str = "") -> str:
    lower = title_str.lower()
    for prefix in FILLER_PREFIXES:
        if lower.startswith(prefix):
            stripped = title_str[len(prefix):].lstrip(' ,.')
            stripped_words = stripped.split()
            # Fallback if result is empty or single short word
            if (not stripped or 
                len(stripped) < 5 or 
                len(stripped_words) < 2):
                if full_response:
                    words = full_response.split()
                    return " ".join(words[:7]) + (
                        "..." if len(words) > 7 else ""
                    )
                return title_str
            return stripped[:1].upper() + stripped[1:]
    return title_str

# Test 1: Standard case - should strip prefix and keep remaining words
res = sanitize_title("Based on the provided data, the PTO policy allows 15 days", "The PTO policy allows 15 days of vacation.")
print("Test 1 Result:", res)
assert res == "The PTO policy allows 15 days"

# Test 2: Single short word "Here" - should fall back to first 7 words of full_response
res = sanitize_title("Based on the provided data, here", "Based on the documents, here is a comprehensive guide to requesting PTO.")
print("Test 2 Result:", res)
assert res == "Based on the documents, here is a..."

# Test 3: Short word with punctuation - should fall back
res = sanitize_title("Based on the documents, Here.", "Here is the information about the holiday schedule.")
print("Test 3 Result:", res)
assert res == "Here is the information about the holiday..."

# Test 4: Less than 5 characters remaining - should fall back
res = sanitize_title("Based on the documents, PTO", "PTO is accrued monthly for full-time employees.")
print("Test 4 Result:", res)
assert res == "PTO is accrued monthly for full-time employees."

print("All sanitization tests passed successfully!")
