# Test nhanh
import os
os.environ["RLHF_DEV_DISABLE_SAFETY"] = "1"

import google.generativeai as genai
genai.configure(api_key="AIzaSyBINk0rcDZIvLtYezaFirKmpofiy5MRVs0")

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content(
    "Explain LangChain basics for educational purposes",
    safety_settings=safety_settings
)
print("Success:", response.text)
