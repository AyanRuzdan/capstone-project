import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter

# Ensure output directory exists
output_dir = './data/plots'
os.makedirs(output_dir, exist_ok=True)

# Load and prepare data
df = pd.read_csv('./data/cluster_assignments.csv')
df['date'] = pd.to_datetime(df['date'])
df['date_only'] = df['date'].dt.date
df['keywords'] = df['keywords'].fillna('')
df['keyword_list'] = df['keywords'].apply(
    lambda x: [kw.strip() for kw in x.split(',') if kw.strip()])

# --- Plot 1: Cluster Trends Over Time ---
cluster_counts_by_date = df.groupby(
    ['date_only', 'cluster']).size().unstack(fill_value=0)

plt.figure(figsize=(12, 6))
ax = cluster_counts_by_date.plot(kind='line', marker='o', ax=plt.gca())
plt.title("Temporal Trend of Cluster Assignments Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Articles in Each Cluster")

# Adjust x-axis ticks to reduce clutter
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))  # Set interval for ticks
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format as 'YYYY-MM-DD'

# Reduce x-tick frequency by setting tick positions explicitly
ticks = ax.get_xticks()
# Only display every 1 out of 3 ticks (about 70% of the time)
ax.set_xticks(ticks[::3])

# Rotate and adjust the x-ticks
plt.xticks(rotation=45, fontsize=8)  # Reduce font size for better fit
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/cluster_trends_over_time.png")
plt.close()

# --- Plot 2: Keyword Trends for Each Cluster ---
keyword_trends = {}
for cluster in sorted(df['cluster'].unique()):
    cluster_df = df[df['cluster'] == cluster]
    keywords_by_date = cluster_df.groupby('date_only')['keyword_list'].apply(
        lambda lists: [kw for sublist in lists for kw in sublist]
    )
    keyword_counts_by_date = keywords_by_date.apply(lambda kws: Counter(kws))
    keyword_counts_df = pd.DataFrame(
        keyword_counts_by_date.tolist(), index=keywords_by_date.index).fillna(0)
    keyword_trends[cluster] = keyword_counts_df

    # --- Plot top keywords for current cluster ---
    keyword_df = keyword_counts_df.sort_index()
    top_keywords = keyword_df.sum().nlargest(5).index
    filtered_df = keyword_df[top_keywords]
    smoothed_df = filtered_df.rolling(window=3, min_periods=1).mean()

    plt.figure(figsize=(14, 6))
    for keyword in smoothed_df.columns:
        plt.plot(smoothed_df.index,
                 smoothed_df[keyword], marker='o', label=keyword)

    plt.title(f"Top Keyword Trends for Cluster {cluster}")
    plt.xlabel("Date")
    plt.ylabel("Keyword Frequency")

    # Adjust x-axis ticks to reduce clutter
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))  # Set interval for ticks
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format as 'YYYY-MM-DD'

    # Reduce x-tick frequency by setting tick positions explicitly
    ticks = plt.gca().get_xticks()
    plt.gca().set_xticks(ticks[::3])  # Only display every 1 out of 3 ticks (about 70% of the time)

    plt.xticks(rotation=45, fontsize=8)  # Reduce font size for better fit
    plt.grid(True)
    plt.legend(title="Keywords", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_{cluster}_keywords.png")
    plt.close()
