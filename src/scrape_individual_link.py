import requests
from bs4 import BeautifulSoup
import time
import csv
import json
from datetime import datetime

base_url = "https://www.hindustantimes.com/science/page-"
headers = {
    'User-Agent': 'Mozilla/5.0'
}


def scrape_ht_science(max_pages=50):
    articles = []

    for page in range(1, max_pages + 1):
        print(f"Scraping page {page}")
        url = base_url + str(page)
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.content, "html.parser")

        cards = soup.find_all("div", class_="cartHolder")

        for card in cards:
            title_tag = card.find("h3")
            link_tag = title_tag.find("a") if title_tag else None
            date_tag = card.find("span", class_="dateTime")

            title = link_tag.text.strip() if link_tag else None
            link = "https://www.hindustantimes.com" + \
                link_tag['href'] if link_tag else None
            pub_date = date_tag.text.strip() if date_tag else None

            if title and link:
                articles.append({
                    "title": title,
                    "link": link,
                })

        time.sleep(1)

    return articles


data = scrape_ht_science(max_pages=50)

with open("./data/ht_science_articles.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(
    f"\nSaved {len(data)} articles to 'ht_science_articles.json'")
