import os
import json
import gzip
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT


# ========== TEXT CLEANING FUNCTION ==========
def clean_text(text):
    """Clean tweets before keyword extraction."""
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove mentions (@username)
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags but keep the word (e.g. #Freedom -> Freedom)
    text = re.sub(r"#", "", text)

    # Remove HTML entities (&amp;, &lt;, etc.)
    text = re.sub(r"&\w+;", "", text)

    # Remove "RT" (retweet markers)
    text = re.sub(r"\brt\b", "", text, flags=re.IGNORECASE)

    # Remove punctuation, numbers, and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Lowercase and strip extra spaces
    text = text.lower().strip()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text


# ========== LOAD POSTS FROM FOLDER ==========
def load_json_posts_from_folder(folder_path):
    """
    Generator that yields posts one by one from .json.gz files.
    Handles line-delimited JSON tweets.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".json.gz"):
            file_path = os.path.join(folder_path, filename)
            try:
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        try:
                            post = json.loads(line.strip())
                            yield post
                        except json.JSONDecodeError as e:
                            print(f"⚠️ Skipped invalid line in {filename}: {e}")
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")


# ========== TF-IDF EXTRACTION ==========
def extract_keywords_tfidf(texts, top_k=10):
    """
    Extract top-k keywords from a list of cleaned texts using TF-IDF.
    """
    texts = [clean_text(text) for text in texts]

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(texts)
    features = vectorizer.get_feature_names_out()

    top_keywords = []
    for i in range(X.shape[0]):
        row = X[i].toarray().flatten()
        top_indices = row.argsort()[-top_k:][::-1]
        keywords = [features[idx] for idx in top_indices if row[idx] > 0]
        top_keywords.append(keywords)

    return top_keywords


# ========== KEYBERT EXTRACTION ==========
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

def extract_keywords_keybert(texts, top_k=10):
    """
    Extract top-k semantic keywords from a list of cleaned texts using KeyBERT.
    """
    keywords = []
    for i, doc in enumerate(texts):
        doc = clean_text(doc)
        if not doc:
            keywords.append([])
            continue
        try:
            result = kw_model.extract_keywords(
                doc,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=top_k
            )
            keywords.append([kw for kw, _ in result])
        except Exception as e:
            print(f"⚠️ KeyBERT failed at index {i}: {e}")
            keywords.append([])
    return keywords


# ========== BATCH PROCESSOR ==========
def process_json_files_in_batches(folder_path, output_csv, screen_names, batch_size=100):
    """
    Process posts in batches and write extracted keyword results to CSV.
    """
    batch = []
    first_write = True

    for post in load_json_posts_from_folder(folder_path):
        user = post.get("user", {})
        screen_name = user.get("screen_name", "").lower().strip()
        if screen_name not in screen_names:
            continue

        message = post.get("full_text") or post.get("text") or ""
        batch.append({
            "screen_name": screen_name,
            "message": message,
            "created_at": post.get("created_at", ""),
            "tweet_id": post.get("id_str") or post.get("id", ""),
        })

        if len(batch) >= batch_size:
            _process_and_save_batch(batch, output_csv, first_write)
            first_write = False
            batch = []

    # Final leftover batch
    if batch:
        _process_and_save_batch(batch, output_csv, first_write)


def _process_and_save_batch(batch, output_csv, first_write):
    df = pd.DataFrame(batch)
    print(f"🧠 Processing batch of {len(df)} tweets...")

    # Extract keywords
    tfidf_keywords = extract_keywords_tfidf(df['message'])
    keybert_keywords = extract_keywords_keybert(df['message'])

    # Determine max number of keywords to use for column expansion
    max_tfidf = max(len(kw_list) for kw_list in tfidf_keywords)
    max_keybert = max(len(kw_list) for kw_list in keybert_keywords)

    # Add expanded keyword columns
    for i in range(max_tfidf):
        df[f'tfidf_kw{i+1}'] = [kw_list[i] if i < len(kw_list) else '' for kw_list in tfidf_keywords]

    for i in range(max_keybert):
        df[f'keybert_kw{i+1}'] = [kw_list[i] if i < len(kw_list) else '' for kw_list in keybert_keywords]

    # OPTIONAL CLEANING STARTS HERE

    # Replace empty keyword strings with NaN for better CSV cleanliness
    df.replace("", pd.NA, inplace=True)

    # Remove rows where both TF-IDF and KeyBERT keywords are all NaN
    df = df[
        df[[f'tfidf_kw{i+1}' for i in range(max_tfidf)]].notna().any(axis=1) |
        df[[f'keybert_kw{i+1}' for i in range(max_keybert)]].notna().any(axis=1)
    ]

    # OPTIONAL CLEANING ENDS HERE

    # Save to CSV
    df.to_csv(output_csv, mode='w' if first_write else 'a', index=False, header=first_write)
    print(f"💾 Saved batch to {output_csv}")

