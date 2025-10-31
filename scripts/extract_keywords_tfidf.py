#!/usr/bin/env python3
import os
import json
import gzip
import re
from typing import List, Dict, Any, Iterable, Set, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================================================
# PROJECT-LEVEL DEFAULTS
# =========================================================
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

DEFAULT_FOLDER = os.path.join(PROJECT_ROOT, "data", "twitter_v1.1")
DEFAULT_USERS_CSV = os.path.join(PROJECT_ROOT, "data", "politician_user_info.csv")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "outputs", "twitter_keywords_extracted_tfidf.csv")

BATCH_SIZE = 50
DEFAULT_TOP_K = 5
TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z0-9]*(?:[-'][a-zA-Z0-9]+)*\b"

# tokens we never really want
DOMAIN_STOP = {
    "people", "news", "today", "time", "year", "years", "day", "days",
    "thing", "things",
    "make", "made", "making",
    "live", "watch", "breaking", "latest", "viral",
    "official", "update", "updates",
    "help", "need", "thank", "thanks", "just", "amp",
    "s", "t",
}

# phrases we’ve actually seen that are boilerplate
PHRASE_STOP = {
    "men women",
    "members veterans",
    "service members",
    "support military",
    "unparalleled support",
    "shown unparalleled",
    "continue work",
    "i'll continue",
    "ll continue",
    "welcome pence",
    "happy birthday",
    "glad",
    "great",
    "enjoyed",
    "proud",
    "administration",   # optional
}

# phrases starting with these are usually not topical
DROP_IF_STARTS = (
    "i ", "i’m ", "i'm ", "i’ll ", "i'll ",
    "we ", "we’re ", "we're ",
    "my ", "our ",
)

# words that are too generic to be good keywords
GENERIC_WORDS = {
    "enjoyed", "enjoy", "incredibly", "incredible",
    "afternoon", "evening", "morning", "today", "tonight",
    "continue", "continuing", "joined", "joining",
    "care", "caring",
    "great", "greatest",
    "happy", "congrats", "congratulations",
    "friend", "family", "families",
    "life", "lives",
    "community", "communities",
    "better", "work", "working",
    "important", "incredible", "incredibly",
    "speak", "speaking",
    "support", "supporting",
    "programs",
    "award",
    "glad",
    "proud",
}

# allow short but meaningful
SHORT_WHITELIST = {"us", "uk", "eu", "nc", "va", "dc", "dod", "gop"}


# =========================================================
# CLEANING
# =========================================================
def clean_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = text.replace("#", " ")

    # collapse possessives / n't so we don't get lone "s"/"t"
    text = re.sub(r"\b([a-zA-Z]+)'s\b", r"\1", text)
    text = re.sub(r"\b([a-zA-Z]+)’s\b", r"\1", text)
    text = re.sub(r"\b([a-zA-Z]+)n't\b", r"\1", text)

    text = re.sub(r"\brt\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9\-\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


# =========================================================
# AUTO-LOAD SCREEN NAMES
# =========================================================
def load_screen_names_from_csv(csv_path: str) -> List[str]:
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found, using fallback users.")
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

    print(f"[WARN] no known twitter column in {csv_path}, using fallback users.")
    return ["example_user"]


# =========================================================
# LOADING TWEETS
# =========================================================
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
# TF-IDF HELPERS
# =========================================================
def _fit_tfidf(texts: List[str], min_df_val: int) -> Optional[TfidfVectorizer]:
    try:
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            token_pattern=TOKEN_PATTERN,
            max_df=0.85,
            min_df=min_df_val,
            strip_accents="unicode",
        )
        X = vec.fit_transform(texts)
        if X.shape[1] == 0:
            return None
        return vec
    except ValueError:
        return None


def _is_informative_phrase(p: str) -> bool:
    """
    Return True if p has at least one token that's *not* in GENERIC_WORDS or DOMAIN_STOP.
    """
    toks = p.split()
    for t in toks:
        if t not in GENERIC_WORDS and t not in DOMAIN_STOP:
            return True
    return False


def dedupe_and_filter(phrases: List[str], max_out: int) -> List[str]:
    cleaned = []
    seen = set()
    for p in phrases:
        p = p.strip()
        if not p:
            continue

        # phrase-level stop
        if p in PHRASE_STOP:
            continue

        lower_p = p.lower()

        # drop procedural starters
        if any(lower_p.startswith(pref) for pref in DROP_IF_STARTS):
            continue

        toks = lower_p.split()

        # drop very short unigrams unless whitelisted
        if len(toks) == 1:
            tok = toks[0]
            if len(tok) <= 2 and tok not in SHORT_WHITELIST:
                continue

        # drop if every word is generic/not topical
        if not _is_informative_phrase(lower_p):
            continue

        if p in seen:
            continue
        seen.add(p)
        cleaned.append(p)

    cleaned.sort(key=lambda x: (-len(x.split()), x))
    return cleaned[:max_out]


def extract_batch_keywords(texts: List[str], top_k: int) -> List[List[str]]:
    cleaned = [clean_text(t) for t in texts]
    vec = _fit_tfidf(cleaned, 2) or _fit_tfidf(cleaned, 1)
    if vec is None:
        return [[] for _ in cleaned]

    X = vec.transform(cleaned)
    feature_names = vec.get_feature_names_out()
    out: List[List[str]] = []

    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            out.append([])
            continue

        idxs = row.indices
        vals = row.data
        order = sorted(range(len(idxs)), key=lambda k: vals[k], reverse=True)
        ranked = [feature_names[idxs[k]] for k in order]
        out.append(dedupe_and_filter(ranked, top_k))

    return out


# =========================================================
# BATCH PROCESSOR
# =========================================================
def process_folder_in_batches(
    folder: str,
    output_csv: str,
    screen_names: Iterable[str],
    batch_size: int = BATCH_SIZE,
    top_k: int = DEFAULT_TOP_K,
) -> None:
    targets: Set[str] = {s.lower().strip() for s in screen_names}
    batch_rows: List[Dict[str, Any]] = []
    wrote_header = False

    for tweet in iter_tweets_from_folder(folder):
        user = tweet.get("user") or {}
        sn = (user.get("screen_name") or "").lower().strip()
        if targets and sn not in targets:
            continue

        message = tweet.get("full_text") or tweet.get("text") or ""
        if is_retweet(tweet, message):
            continue

        batch_rows.append(
            {
                "screen_name": sn,
                "message": message,
                "created_at": tweet.get("created_at", ""),
                "tweet_id": tweet.get("id_str") or tweet.get("id") or "",
            }
        )

        if len(batch_rows) >= batch_size:
            _process_and_append(batch_rows, output_csv, wrote_header, top_k)
            wrote_header = True
            batch_rows = []

    if batch_rows:
        _process_and_append(batch_rows, output_csv, wrote_header, top_k)


def _process_and_append(
    rows: List[Dict[str, Any]],
    output_csv: str,
    wrote_header: bool,
    top_k: int,
) -> None:
    df = pd.DataFrame(rows)
    print(f"[INFO] processing batch of {len(df)} tweets...")

    kw_lists = extract_batch_keywords(df["message"].tolist(), top_k=top_k)

    max_k = max((len(kws) for kws in kw_lists), default=0)
    for i in range(max_k):
        df[f"tfidf_kw{i+1}"] = [kws[i] if i < len(kws) else "" for kws in kw_lists]

    # drop rows with no keywords at all
    if max_k > 0:
        kw_cols = [f"tfidf_kw{i+1}" for i in range(max_k)]
        df = df[df[kw_cols].notna().any(axis=1)]

    df.replace("", pd.NA, inplace=True)

    mode = "a" if wrote_header else "w"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, mode=mode, index=False, header=not wrote_header)
    print(f"[INFO] wrote {len(df)} rows to {output_csv}")


# =========================================================
# MAIN
# =========================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        "TF-IDF tweet keywords (batched)",
        description="Runs with hard-coded defaults if you don't provide flags.",
        add_help=True,
    )
    parser.add_argument("--folder", help="folder with .json.gz files")
    parser.add_argument("--out", help="output CSV path")
    parser.add_argument("--users", nargs="+", help="screen names to keep (no @)")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
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
    print(f"[INFO] batch size: {args.batch}")

    process_folder_in_batches(
        folder=folder,
        output_csv=out_path,
        screen_names=users,
        batch_size=args.batch,
        top_k=args.topk,
    )


if __name__ == "__main__":
    main()
