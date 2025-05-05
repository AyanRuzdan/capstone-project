import requests
import os
from serpapi import GoogleSearch
from dotenv import load_dotenv
load_dotenv()


def search_web(query):
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERP_API_KEY")
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("organic_results", [])


def get_gemini_response(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Error: {response.status_code} - {response.text}"


def chat():
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        search_results = search_web(user_input)
        context = ""
        for result in search_results[:3]:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            context += f"{title}: {snippet}\n"
        prompt = f"User asked: {user_input}\nRelevant information:\n{context}\nProvide a helpful response."
        response = get_gemini_response(prompt)
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    chat()
