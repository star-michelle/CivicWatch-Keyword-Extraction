# === main.py ===
import pandas as pd
import os
from scripts.extract_keywords import process_json_files_in_batches

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

# Load politician metadata
politicians = pd.read_csv("data/politician_user_info.csv")
politicians['screen_name'] = politicians['screen_name'].str.lower().str.strip()

# Filter only Twitter users
twitter_pols = politicians[politicians['platform'].str.lower() == 'twitter'].copy()
twitter_screen_names = twitter_pols['screen_name'].astype(str).str.strip().str.lower().tolist()

# === Process Twitter Data in Batches ===
print("📦 Processing Twitter data in batches...")

process_json_files_in_batches(
    folder_path="data/twitter_v1.1",
    output_csv="outputs/twitter_keywords_extracted.csv",
    screen_names=twitter_screen_names,
    batch_size=1000
)

print("✅ Done!")
