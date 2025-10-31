#!/usr/bin/env python3
"""
RAKE-based keyword extraction for tweets (batched, tweet-tuned).

- input folder  : ../data/twitter_v1.1
- users csv     : ../data/politician_user_info.csv
- output csv    : ../outputs/twitter_keywords_extracted_rake.csv
- batch size    : 50
- top_k         : 5

Changes from vanilla RAKE:
- cap phrase length (MAX_PHRASE_WORDS) so we don't get 25-word blobs
- drop emoji / arrow / numeric-only phrases
- bigger politician/tweet stopword list
- post-filter to kill 1-word sentiment / glue
"""

import os
import json
import gzip
import re
from typing import List, Dict, Any, Iterable, Set, Tuple

import pandas as pd

# =========================================================
# PROJECT DEFAULTS
# =========================================================
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

DEFAULT_FOLDER = os.path.join(PROJECT_ROOT, "data", "twitter_v1.1")
DEFAULT_USERS_CSV = os.path.join(PROJECT_ROOT, "data", "politician_user_info.csv")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "outputs", "twitter_keywords_extracted_rake.csv")

BATCH_SIZE = 50
TOP_K = 5

# max words we allow in a single RAKE phrase (tweets are short!)
MAX_PHRASE_WORDS = 6

# =========================================================
# CLEANING
# =========================================================
URL_RE = re.compile(r"http\S+|www\S+|https\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+", re.IGNORECASE)
HTML_RE = re.compile(r"&\w+;", re.IGNORECASE)

def clean_tweet(text: str) -> str:
    """Light-clean a tweet but keep informative tokens."""
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    # turn hashtags into words
    text = text.replace("#", " ")
    # normalize whitespace + lowercase
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

# =========================================================
# STOPWORDS for RAKE
# =========================================================
# core english + social + political boilerplate
RAKE_STOPWORDS = {
    # english function words
    "a","an","the","and","or","of","for","to","in","on","at","with","is","are","was","were",
    "be","been","by","from","this","that","it","its","as","about","into","over","after","before",
    "we","i","you","they","them","our","us","my","your","their","his","her","him",
    # tweet filler
    "today","tonight","yesterday","tomorrow",
    "please","share","tune","retweet",
    "here","there","up","out","back",
    # congrats / sentiment / politician niceness
    "congratulations","congrats","happy","proud","honor","honored","humbled","thrilled",
    "great","incredible","incredibly","awesome","amazing","wonderful","enjoyed","glad",
    "thanks","thank","thanking",
    # politician yapping
    "support","supported","supporting","supporters",
    "work","worked","working","continue","continued","continuing",
    "join","joined","joining","joined",
    "welcome","welcomed","welcoming",
    "saying","says","said",
    # your boilerplate you said you don't want
    "members","member","military","service","services","veteran","veterans",
    "families","family","community","communities","colleagues","administration",
    # linking words we don't want as RAKE heads
    "but","so","because","while","when","then","than","if","also",
    "have","has","had","will","would","can","could","should","must","may",
    "just"
}

# split on punctuation
PUNCT_SPLIT_RE = re.compile(r"[,.!?;:()\-–—/]+")

# emojis / arrows / flags we want to drop if a phrase ends with them
EMOJI_TAIL_RE = re.compile(r"[\u2190-\u21ff\u2190-\u21ff\u2600-\u27bf\ufe0f]+$")
BAD_STARTS = {
    "teeth", "beautiful", "because", "have", "has", "will", "can’t", "cant",
    "we’re", "were", "i’m", "im", "i was", "i will", "i’ll", "i am",
    "these", "those", "this is", "it was", "it’s", "its",
}

def starts_bad(p: str) -> bool:
    low = p.lower()
    for s in BAD_STARTS:
        if low.startswith(s):
            return True
    return False

CLAUSE_SPLIT_RE = re.compile(r"\b(and|but|that|which|because|while|when|who|whom|whose)\b")

def split_clauses(p: str) -> list[str]:
    # split into smaller candidate phrases
    parts = CLAUSE_SPLIT_RE.split(p)
    # parts is like ['main', 'and', 'rest', 'but', 'rest2' ...]
    out = []
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip()
        if chunk:
            out.append(chunk)
    return out

def post_filter_phrases(phrases: list[str]) -> list[str]:
    cleaned = []
    for p in phrases:
        if is_bad_phrase(p):
            continue

        # break long-ish ones into clauses, then evaluate
        subparts = split_clauses(p)
        for sub in subparts:
            if is_bad_phrase(sub):
                continue
            if starts_bad(sub):
                continue
            words = sub.split()
            if len(words) > 1:
                cleaned.append(sub)
            else:
                # 1-word: keep only if looks like name / acronym
                if sub.istitle() or sub.isupper():
                    cleaned.append(sub)
    return cleaned

def is_bad_phrase(p: str) -> bool:
    """Filter clearly bad / noisy phrases after RAKE."""
    p = p.strip()
    if not p:
        return True
    # drop emoji / arrow / flag endings
    if EMOJI_TAIL_RE.search(p):
        return True
    # drop the down-arrow tweet guide
    if p.endswith("⬇️") or p.endswith("⬇️⬇️") or "⬇️⬇️⬇️" in p:
        return True
    # drop pure numbers
    if re.fullmatch(r"[0-9 ]+", p):
        return True
    # drop single punctuation leftovers
    if re.fullmatch(r"[-–—]+", p):
        return True
    return False

def post_filter_phrases(phrases: List[str]) -> List[str]:
    """Final clean step: drop 1-word fluff, keep entities."""
    out = []
    for p in phrases:
        if is_bad_phrase(p):
            continue
        words = p.split()
        # keep multiword, they're usually more informative
        if len(words) > 1:
            out.append(p)
            continue
        # len == 1: keep only if looks like a proper name / acronym
        w = words[0]
        if w.istitle() or w.isupper():
            out.append(p)
            continue
        # else drop 1-word generic
    return out

# =========================================================
# RAKE CORE
# =========================================================
def rake_extract(text: str, top_k: int = TOP_K) -> List[str]:
    """
    Lightweight RAKE tuned for tweets.
    1. split into candidate phrases by stopwords/punct
    2. score words by (degree / freq)
    3. phrase score = sum(word scores)
    4. return top_k phrases (post-filtered)
    """
    if not text:
        return []

    # split by punctuation first to kill urls leftover
    text_punct_spl = PUNCT_SPLIT_RE.sub(" ", text)
    words = text_punct_spl.split()

    phrases: List[List[str]] = []
    current: List[str] = []

    for w in words:
        wl = w.strip().lower()
        if not wl:
            continue
        if wl in RAKE_STOPWORDS:
            if current:
                phrases.append(current)
                current = []
        else:
            current.append(wl)
    if current:
        phrases.append(current)

    if not phrases:
        return []

    # CAP phrase length to avoid monsters
    capped_phrases: List[List[str]] = []
    for ph in phrases:
        if len(ph) > MAX_PHRASE_WORDS:
            ph = ph[:MAX_PHRASE_WORDS]
        capped_phrases.append(ph)

    # ---- word stats
    word_freq: Dict[str, int] = {}
    word_degree: Dict[str, int] = {}
    for phrase in capped_phrases:
        degree = len(phrase) - 1
        for w in phrase:
            word_freq[w] = word_freq.get(w, 0) + 1
            word_degree[w] = word_degree.get(w, 0) + degree

    for w in word_freq:
        word_degree[w] = word_degree[w] + word_freq[w]  # degree += freq

    word_score = {w: (word_degree[w] / word_freq[w]) for w in word_freq}

    # ---- phrase scores
    scored: List[Tuple[str, float]] = []
    for phrase in capped_phrases:
        phrase_clean = " ".join(phrase).strip()
        if not phrase_clean:
            continue
        score = sum(word_score[w] for w in phrase)
        scored.append((phrase_clean, score))

    # sort by score desc
    scored.sort(key=lambda x: x[1], reverse=True)

    # dedupe and cap
    seen = set()
    raw_out: List[str] = []
    for p, _ in scored:
        if p in seen:
            continue
        seen.add(p)
        raw_out.append(p)
        if len(raw_out) >= top_k * 2:
            # collect a bit more, we'll filter next
            break

    # post-filter
    filtered = post_filter_phrases(raw_out)

    # final top_k
    return filtered[:top_k]

# =========================================================
# LOADING
# =========================================================
def load_screen_names_from_csv(csv_path: str) -> List[str]:
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found. Using fallback user.")
        return ["example_user"]
    df = pd.read_csv(csv_path)
    candidates = ["screen_name", "twitter_handle", "twitter", "handle"]
    for col in candidates:
        if col in df.columns:
            names = (
                df[col]
                .dropna()
                .astype(str)
                .str.strip()
                .str.replace("@", "", regex=False)
                .tolist()
            )
            if names:
                print(f"[INFO] loaded {len(names)} users from {col} in {csv_path}")
                return names
    print(f"[WARN] no known twitter column in {csv_path}. Using fallback.")
    return ["example_user"]

def iter_tweets_from_folder(folder: str) -> Iterable[Dict[str, Any]]:
    for name in os.listdir(folder):
        if not name.endswith(".json.gz"):
            continue
        path = os.path.join(folder, name)
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[WARN] can't read {path}: {e}")

def is_retweet(tweet: Dict[str, Any], message: str) -> bool:
    if "retweeted_status" in tweet:
        return True
    if message.lower().startswith("rt "):
        return True
    return False

# =========================================================
# PROCESSING
# =========================================================
def process_folder_rake(
    folder: str,
    output_csv: str,
    screen_names: Iterable[str],
    batch_size: int = BATCH_SIZE,
    top_k: int = TOP_K,
) -> None:
    targets: Set[str] = {s.lower().strip() for s in screen_names}
    rows_buffer: List[Dict[str, Any]] = []
    wrote_header = False

    for tweet in iter_tweets_from_folder(folder):
        user = tweet.get("user") or {}
        sn = (user.get("screen_name") or "").lower().strip()
        if targets and sn not in targets:
            continue

        message = tweet.get("full_text") or tweet.get("text") or ""
        if is_retweet(tweet, message):
            continue

        rows_buffer.append({
            "screen_name": sn,
            "message": message,
            "created_at": tweet.get("created_at", ""),
            "tweet_id": tweet.get("id_str") or tweet.get("id") or "",
        })

        if len(rows_buffer) >= batch_size:
            _process_batch_rake(rows_buffer, output_csv, wrote_header, top_k)
            wrote_header = True
            rows_buffer = []

    if rows_buffer:
        _process_batch_rake(rows_buffer, output_csv, wrote_header, top_k)

def _process_batch_rake(
    rows: List[Dict[str, Any]],
    output_csv: str,
    wrote_header: bool,
    top_k: int,
) -> None:
    print(f"[INFO] processing batch of {len(rows)} tweets with RAKE ...")
    df = pd.DataFrame(rows)

    kw_lists: List[List[str]] = []
    for msg in df["message"].tolist():
        cleaned = clean_tweet(msg)
        kws = rake_extract(cleaned, top_k=top_k)
        kw_lists.append(kws)

    max_k = max((len(k) for k in kw_lists), default=0)
    for i in range(max_k):
        df[f"rake_kw{i+1}"] = [k[i] if i < len(k) else "" for k in kw_lists]

    # keep all rows, but NaN out empties for cleanliness
    df.replace("", pd.NA, inplace=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    mode = "a" if wrote_header else "w"
    df.to_csv(output_csv, mode=mode, index=False, header=not wrote_header)
    print(f"[INFO] wrote {len(df)} rows to {output_csv}")

# =========================================================
# CLI
# =========================================================
def main():
    import argparse
    parser = argparse.ArgumentParser("RAKE tweet keywords (batched, tweet-tuned)")
    parser.add_argument("--folder", help="folder with .json.gz files")
    parser.add_argument("--out", help="output CSV path")
    parser.add_argument("--users", nargs="+", help="screen names (no @)")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--topk", type=int, default=TOP_K)
    args = parser.parse_args()

    folder = args.folder or DEFAULT_FOLDER
    out_path = args.out or DEFAULT_OUTPUT

    if args.users:
        users = args.users
    else:
        users = load_screen_names_from_csv(DEFAULT_USERS_CSV)

    print(f"[INFO] using folder: {folder}")
    print(f"[INFO] writing to: {out_path}")
    print(f"[INFO] users: {users[:10]}{' ...' if len(users) > 10 else ''}")

    process_folder_rake(
        folder=folder,
        output_csv=out_path,
        screen_names=users,
        batch_size=args.batch,
        top_k=args.topk,
    )

if __name__ == "__main__":
    main()
