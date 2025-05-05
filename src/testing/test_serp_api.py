import os
from serpapi import GoogleSearch
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("SERP_API_KEY")


def search_google(query):
    params = {
        "q": query,
        "api_key": API_KEY,
        "engine": 'google',
        "num": 5
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    for result in results.get("organic_results", []):
        print("🔹", result.get("title"))
        print("📎", result.get("link"))
        print("📝", result.get("snippet"))
        print("-" * 60)


if __name__ == "__main__":
    search_google("latest news in science")
