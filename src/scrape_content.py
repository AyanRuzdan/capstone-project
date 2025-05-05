import requests
from bs4 import BeautifulSoup
import json
import time


def get_body(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch {url} — {e}")
        return None, None

    soup = BeautifulSoup(res.text, "html.parser")
    scripts = soup.find_all('script', type='application/ld+json')

    for script in scripts:
        try:
            data = json.loads(script.string)
            if isinstance(data, list):
                for item in data:
                    if 'articleBody' in item and 'datePublished' in item:
                        return item['datePublished'], item['articleBody']
            elif 'articleBody' in data and 'datePublished' in data:
                return data['datePublished'], data['articleBody']
        except (json.JSONDecodeError, TypeError):
            continue

    return None, None


def extract_and_save(json_file, output_file="data/scraped_articles.jsonl"):
    with open(json_file, "r") as file:
        data = json.load(file)

    saved_articles = 0
    with open(output_file, "w", encoding="utf-8") as outfile:
        for item in data:
            url = item.get('link')
            title = item.get('title', 'No Title')
            if not url:
                print("[WARN] No URL found for an item. Skipping.")
                continue

            date_published, article_body = get_body(url)
            if date_published and article_body:
                article = {
                    "title": title,
                    "url": url,
                    "date": date_published,
                    "text": article_body
                }
                json.dump(article, outfile, ensure_ascii=False)
                outfile.write("\n")
                saved_articles += 1
                print(f"[SAVED] {title}")
            else:
                print(f"[SKIPPED] No content for {url}")

    print(f"\n✅ Done. {saved_articles} articles saved to '{output_file}'.")


# Run it
extract_and_save("data/ht_science_articles.json")
