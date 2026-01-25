"""
List available Gemini models. Requires GEMINI_API_KEY in environment.
"""
import os
import sys

api_key = os.environ.get("GEMINI_API_KEY", "").strip()
if not api_key:
    print("GEMINI_API_KEY environment variable is required.", file=sys.stderr)
    sys.exit(1)

import google.generativeai as genai

genai.configure(api_key=api_key)
print("Available Gemini models:")
for m in genai.list_models():
    if "gemini" in m.name:
        print(f"  - {m.name}")
