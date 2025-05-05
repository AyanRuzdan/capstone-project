import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
# Updated URL for a text-focused, efficient model
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

print(API_KEY)

params = {
    "key": API_KEY
}

headers = {
    "Content-Type": "application/json",
}

data = {
    "contents": [
        {
            "parts": [
                {
                    "text": "What happened to Chandrayaan-3 on August 23, 2023?"
                }
            ]
        }
    ]
}

response = requests.post(API_URL, params=params, headers=headers, json=data)

if response.status_code == 200:
    result = response.json()
    print("Response:", result)
else:
    print(
        f"Request failed with status code {response.status_code}: {response.text}")
