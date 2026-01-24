# check_models.py
import google.generativeai as genai
import os

# Set your key directly or load from environment
os.environ["GEMINI_API_KEY"] = "AIzaSyA7mL0_exRI0t0LtPjOzpveQHzn6UTGvbg"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

print("Available Gemini Models:")
for m in genai.list_models():
    if 'gemini' in m.name:
        print(f"- {m.name} (Methods: {m.supported_generation_methods})")	