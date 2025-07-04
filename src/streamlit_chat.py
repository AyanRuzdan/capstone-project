from PIL import Image
import streamlit as st
import requests
import os
import subprocess
import json
from serpapi import GoogleSearch
from dotenv import load_dotenv
import time

load_dotenv()


def search_web(query):
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERP_API_KEY")
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    print("RESULTS", results['organic_results'])
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


st.set_page_config(page_title="News ChatBot", layout="wide")

with st.sidebar:
    st.title("AI News Assistant")
    tab = st.radio("Select Tab", [
                   "News ChatBot", "Fetch news links", "Fetch news content", "Perform Clustering", "Temporal Trend"])

if tab == "News ChatBot":
    st.header("News ChatBot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about recent news..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            timer_placeholder = st.empty()
            timer_placeholder.caption(
                "Searching web and generating response...")

            start_time = time.time()
            search_results = search_web(prompt)
            context = ""
            for result in search_results[:3]:
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                context += f"{title}: {snippet}\n"

            chat_history = ""
            for message in st.session_state.messages:
                role = "User" if message["role"] == "user" else "Assistant"
                chat_history += f"{role}: {message['content']}\n"
            print(chat_history)
            constructed_prompt = (
                f"{chat_history}"
                f"User asked: {prompt}\n"
                f"Relevant web info:\n{context}\n"
                f"Provide a helpful, contextual answer in a conversational tone."
            )

            response = get_gemini_response(constructed_prompt)
            duration = round(time.time() - start_time, 2)

            response_placeholder.markdown(response)
            timer_placeholder.caption(f"Response time: {duration} seconds")

        st.session_state.messages.append(
            {"role": "assistant", "content": response})
elif tab == "Fetch news links":
    st.header("Science News Scraper")

    json_path = "data/ht_science_articles.json"
    if os.path.exists(json_path):
        st.subheader("Scraped Articles (JSON Preview)")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                if isinstance(data, list) and len(data) > 0 and "title" in data[0]:
                    st.write(f"Total Articles: {len(data)}")
                    st.json(data[:10])  # Show only top 10 articles
                    st.download_button(
                        label="Download full JSON",
                        data=json.dumps(data, indent=2),
                        file_name="ht_science_articles.json",
                        mime="application/json"
                    )
                else:
                    st.warning(
                        "The file exists but doesn't contain valid article data.")
        except json.JSONDecodeError:
            st.error(
                "Could not parse the file as JSON. It may be corrupted or contain HTML.")
    else:
        st.info("No JSON file found yet. Click the button below to fetch articles.")

    if st.button("Fetch all links"):
        with st.spinner("Scraping science articles... please wait."):
            result = subprocess.run(
                ["python", "src/scrape_individual_link.py"], capture_output=True, text=True)

            if result.returncode == 0:
                st.success("Scraping complete!")
                st.text(result.stdout)
            else:
                st.error("Error occurred during scraping.")
                st.text(result.stderr)

elif tab == "Fetch news content":
    st.header("Fetch News Content")
    json_path = "data/scraped_articles.jsonl"

    if os.path.exists(json_path):
        st.subheader("Scraped Articles (JSONL Preview)")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                for line in f:
                    st.json(json.loads(line))
                    break
            with open(json_path, "r", encoding="utf-8") as f:
                total_lines = sum(1 for _ in f)
            st.caption(f"Total articles: {total_lines}")
        except json.JSONDecodeError:
            st.error(
                "Could not parse the file as JSON. It may be corrupted or contain HTML.")
    else:
        st.info("No JSONL file found yet. Click the button below to fetch articles.")

    if st.button("Fetch all content"):
        with st.spinner("Fetching article content... please wait."):
            result = subprocess.run(
                ["python", "src/scrape_content.py"], capture_output=True, text=True)

            if result.returncode == 0:
                st.success("Content fetching complete!")
                st.text(result.stdout)
            else:
                st.error("Error occurred during content fetching.")
                st.text(result.stderr)

elif tab == "Perform Clustering":
    st.header("Perform Article Clustering")

    plot_path = "data/cluster_plot.png"

    # Show plot if already generated
    if os.path.exists(plot_path):
        st.subheader("Latest Clustering Plot")
        with open(plot_path, "rb") as f:
            img_bytes = f.read()
        st.image(img_bytes, caption="Clustering Visualization", width=1100)
    else:
        st.info("No clustering plot found yet. Run clustering to generate one.")

    if st.button("Run Clustering"):
        with st.spinner("Clustering articles... please wait."):
            result = subprocess.run(
                ["python", "src/clustering.py"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                st.success("Clustering complete!")
                st.text(result.stdout)

                if os.path.exists(plot_path):
                    with open(plot_path, "rb") as f:
                        img_bytes = f.read()
                    st.image(
                        img_bytes, caption="Updated Clustering Visualization", use_column_width=True)
                else:
                    st.warning("Clustering finished, but no plot was found.")
            else:
                st.error("Clustering failed.")
                st.text(result.stderr)
elif tab == "Temporal Trend":
    st.header("Temporal Trend of Clusters")

    if st.button("Run Temporal Trend Analysis"):
        with st.spinner("Generating temporal trend plots... please wait."):
            result = subprocess.run(
                ["python", "src/temporal_trend.py"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                st.success("Temporal trend analysis complete!")
                st.text(result.stdout)
            else:
                st.error("Error occurred during trend analysis.")
                st.text(result.stderr)

    plot_dir = "data/plots"
    if os.path.exists(plot_dir):
        plot_files = sorted(
            [f for f in os.listdir(plot_dir) if f.startswith(
                "cluster_") and f.endswith(".png")]
        )

        if plot_files:
            st.subheader("Cluster-wise Temporal Trend Carousel")

            selected = st.slider("Select Cluster Plot",
                                 0, len(plot_files)-1, 0)
            selected_plot = plot_files[selected]
            img_path = os.path.join(plot_dir, selected_plot)
            st.image(img_path, caption=selected_plot.replace(
                "_", " ").replace(".png", ""), width=1200)
        else:
            st.info(
                "No plots found. Please generate them by clicking the button above.")
    else:
        st.info("Plot directory not found. Run the script to generate plots.")
