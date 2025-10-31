import os
import json
import gzip
import re
import pandas as pd
from typing import List, Iterable, Dict, Any, Optional

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keybert import KeyBERT


# =========================
# CONFIG
# =========================
# Default number of keywords PER METHOD (each row gets up to this many TF-IDF + KeyBERT keywords)
DEFAULT_TOP_K = 5

# Allow hyphenated & numbered tokens like "covid-19", keep simple apostrophes.
TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z0-9]*(?:[-'][a-zA-Z0-9]+)*\b"

# Some very generic words that slip past stopwords for tweets
DOMAIN_STOP = {
    "people", "news", "today", "time", "year", "years", "day", "days", "thing", "things",
    "make", "made", "making", "tweet", "thread", "video", "videos", "live", "watch",
    "breaking", "latest", "viral", "official", "update", "updates"
}


# =========================
# CLEANING
# =========================
def clean_text(text: Any) -> str:
    """Light-clean text but preserve hyphens/numbers so 'covid-19' and names survive."""
    if not isinstance(text, str):
        return ""
    # Remove URLs, mentions, & HTML entities
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    # Keep hashtag words (remove # only)
    text = re.sub(r"#", "", text)
    # Remove RT markers
    text = re.sub(r"\brt\b", " ", text, flags=re.IGNORECASE)
    # Keep letters, digits, hyphens, apostrophes, and spaces
    text = re.sub(r"[^A-Za-z0-9\-\s']", " ", text)
    # Normalize whitespace & case
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


# =========================
# LOADING
# =========================
def load_json_posts_from_folder(folder_path: str) -> Iterable[Dict[str, Any]]:
    """
    Generator that yields posts one by one from .json.gz files.
    Handles line-delimited JSON tweets.
    """
    for filename in os.listdir(folder_path):
        if not filename.endswith(".json.gz"):
            continue
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


# =========================
# DEDUPING & NORMALIZATION
# =========================
def _normalize_phrase(phrase: str) -> str:
    # unify whitespace/hyphen variants: "covid 19" ~= "covid-19"
    p = phrase.lower().strip()
    p = re.sub(r"\s+", " ", p)
    p = p.replace("’", "'")
    # map common variants for subsumption checks
    p = p.replace("-19", " 19")
    return p


def dedupe_phrases(candidates: List[str], max_out: int) -> List[str]:
    """Remove exact dupes, subset phrases, and overly generic tokens. Prefer bigrams."""
    seen = set()
    cleaned = []
    for c in candidates:
        c = c.strip()
        if not c or c in seen:
            continue
        toks = c.split()
        if any(t in DOMAIN_STOP for t in toks):
            continue
        if len(toks) == 1 and (len(toks[0]) <= 2):
            # keep common short exceptions
            if toks[0] not in {"us", "uk", "eu", "ai"}:
                continue
        seen.add(c)
        cleaned.append(c)

    # prefer more specific phrases first
    cleaned.sort(key=lambda x: (-len(x.split()), x))

    final = []
    finals_norm = []
    for c in cleaned:
        cn = _normalize_phrase(c)
        if any(cn in fn or fn in cn for fn in finals_norm):  # containment either way
            continue
        finals_norm.append(cn)
        final.append(c)
        if len(final) >= max_out:
            break
    return final


# =========================
# KEYWORD EXTRACTION — TF-IDF
# =========================
def extract_keywords_tfidf(texts: Iterable[str], top_k: int = DEFAULT_TOP_K) -> List[List[str]]:
    """
    Up to top_k salient (1-2)-gram phrases per doc, preferring bigrams over overlapping unigrams.
    Uses min_df=2 to reduce one-off noise; falls back to min_df=1 if needed.
    """
    texts = [clean_text(t) for t in texts]

    def _fit_vectorizer(min_df_val: int) -> Optional[TfidfVectorizer]:
        try:
            v = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                token_pattern=TOKEN_PATTERN,
                max_df=0.85,
                min_df=min_df_val,
                strip_accents="unicode"
            )
            X = v.fit_transform(texts)
            if X.shape[1] == 0:
                return None
            return v
        except ValueError:
            return None

    # try stricter, then relax if nothing survives
    vectorizer = _fit_vectorizer(min_df_val=2) or _fit_vectorizer(min_df_val=1)
    if vectorizer is None:
        return [[] for _ in texts]

    X = vectorizer.transform(texts)
    features = vectorizer.get_feature_names_out()
    features_list = list(features)

    out: List[List[str]] = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            out.append([])
            continue
        idxs = row.indices
        vals = row.data
        # sort by score high->low
        order = sorted(range(len(idxs)), key=lambda k: vals[k], reverse=True)
        ranked = [features_list[idxs[k]] for k in order]
        out.append(dedupe_phrases(ranked, top_k))
    return out


# =========================
# KEYWORD EXTRACTION — KEYBERT
# =========================
kw_model = KeyBERT(model="all-MiniLM-L6-v2")

def extract_keywords_keybert(texts: Iterable[str], top_k: int = DEFAULT_TOP_K) -> List[List[str]]:
    """
    KeyBERT with MMR and 1-2 grams; de-dup and prefer specific phrases.
    Over-asks (2x) then dedupes down to top_k.
    """
    candidate_vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=TOKEN_PATTERN,
        lowercase=True,
        min_df=1
    )

    keywords: List[List[str]] = []
    for i, doc in enumerate(texts):
        doc = clean_text(doc)
        if not doc:
            keywords.append([])
            continue
        try:
            result = kw_model.extract_keywords(
                doc,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                use_mmr=True,
                diversity=0.7,
                top_n=top_k * 2,
                vectorizer=candidate_vectorizer
            )
            ranked = [kw for kw, _ in result]
            keywords.append(dedupe_phrases(ranked, top_k))
        except Exception as e:
            print(f"⚠️ KeyBERT failed at index {i}: {e}")
            keywords.append([])
    return keywords


# =========================
# BATCH PROCESSOR
# =========================
def process_json_files_in_batches(
    folder_path: str,
    output_csv: str,
    screen_names: Iterable[str],
    batch_size: int = 100,
    top_k: int = DEFAULT_TOP_K
) -> None:
    """
    Process posts in batches and write extracted keyword results to CSV.
    """
    normalized_targets = {s.lower().strip() for s in screen_names}
    batch: List[Dict[str, Any]] = []
    first_write = True

    for post in load_json_posts_from_folder(folder_path):
        user = post.get("user", {})
        screen_name = (user.get("screen_name") or "").lower().strip()
        if screen_name not in normalized_targets:
            continue

        message = post.get("full_text") or post.get("text") or ""
        batch.append({
            "screen_name": screen_name,
            "message": message,
            "created_at": post.get("created_at", ""),
            "tweet_id": post.get("id_str") or post.get("id", ""),
        })

        if len(batch) >= batch_size:
            _process_and_save_batch(batch, output_csv, first_write, top_k)
            first_write = False
            batch = []

    # Final leftover batch
    if batch:
        _process_and_save_batch(batch, output_csv, first_write, top_k)


def _process_and_save_batch(
    batch: List[Dict[str, Any]],
    output_csv: str,
    first_write: bool,
    top_k: int
) -> None:
    df = pd.DataFrame(batch)
    print(f"🧠 Processing batch of {len(df)} tweets...")

    # Extract keywords
    tfidf_keywords = extract_keywords_tfidf(df['message'], top_k=top_k)
    keybert_keywords = extract_keywords_keybert(df['message'], top_k=top_k)

    # Determine max number of keywords to use for column expansion
    max_tfidf = max((len(kw_list) for kw_list in tfidf_keywords), default=0)
    max_keybert = max((len(kw_list) for kw_list in keybert_keywords), default=0)

    # Add expanded keyword columns
    for i in range(max_tfidf):
        df[f'tfidf_kw{i+1}'] = [kw_list[i] if i < len(kw_list) else '' for kw_list in tfidf_keywords]

    for i in range(max_keybert):
        df[f'keybert_kw{i+1}'] = [kw_list[i] if i < len(kw_list) else '' for kw_list in keybert_keywords]

    # Replace empty keyword strings with NaN for better CSV cleanliness
    df.replace("", pd.NA, inplace=True)

    # Remove rows where both TF-IDF and KeyBERT keywords are all NaN
    tfidf_cols = [f'tfidf_kw{i+1}' for i in range(max_tfidf)]
    keybert_cols = [f'keybert_kw{i+1}' for i in range(max_keybert)]
    if tfidf_cols or keybert_cols:
        df = df[
            (df[tfidf_cols].notna().any(axis=1) if tfidf_cols else False) |
            (df[keybert_cols].notna().any(axis=1) if keybert_cols else False)
        ]

    # Save to CSV
    df.to_csv(output_csv, mode='w' if first_write else 'a', index=False, header=first_write)
    print(f"💾 Saved batch to {output_csv}")


# =========================
# OPTIONAL: quick manual run
# =========================
if __name__ == "__main__":
    """
    Example usage:

    process_json_files_in_batches(
        folder_path="/path/to/folder/with/json.gz",
        output_csv="keywords_out.csv",
        screen_names={"account1", "account2"},
        batch_size=200,
        top_k=5
    )
    """
    pass
