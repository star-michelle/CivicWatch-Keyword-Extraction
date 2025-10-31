#!/usr/bin/env python3
"""
Ollama keyword extraction (STRICT, no made-up keywords).

How it works:
1. Read tweets from ../data/twitter_v1.1
2. Filter by users from ../data/politician_user_info.csv
3. For each tweet:
   - clean text (remove URLs, @mentions, #)
   - ask Ollama for keyword phrases
   - **keep ONLY keywords that are textually present in the cleaned tweet**
4. Write to ../outputs/twitter_keywords_extracted_ollama_strict.csv

If Ollama hallucinates, we drop that item.
If it hallucinates all of them, the row has no keywords and we drop that row.
"""

import os
import json
import gzip
import re
import time
from typing import List, Dict, Any, Iterable, Optional, Set

import requests
import pandas as pd

# =========================================================
# PROJECT DEFAULTS (your layout)
# =========================================================
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

DEFAULT_FOLDER = os.path.join(PROJECT_ROOT, "data", "twitter_v1.1")
DEFAULT_USERS_CSV = os.path.join(PROJECT_ROOT, "data", "politician_user_info.csv")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "outputs", "twitter_keywords_extracted_ollama_strict.csv")

DEFAULT_BATCH_SIZE = 50
DEFAULT_TOP_K = 5
DEFAULT_OLLAMA_MODEL = "llama3"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

# =========================================================
# LIGHT CLEAN
# =========================================================
URL_RE = re.compile(r"http\S+|www\S+|https\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+", re.IGNORECASE)
HTML_RE = re.compile(r"&\w+;", re.IGNORECASE)

def light_clean(text: str) -> str:
    """Remove stuff the LLM shouldn't see but keep main words."""
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# a second normalization for the STRICT check (lowercase + collapse spaces)
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()

# =========================================================
# LOAD USERS
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

# =========================================================
# TWEET ITERATOR
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
# OLLAMA CALL (STRICT PROMPT)
# =========================================================
PROMPT_TEMPLATE = """You will extract keyword PHRASES from ONE social media post.

RULES (IMPORTANT):
1. USE ONLY WORDS AND PHRASES THAT ACTUALLY APPEAR IN THE POST.
2. Do NOT invent bills, names, organizations, or places.
3. Prefer specific entities (people, places, orgs, bills, events) that appear in the text.
4. Each item 1-4 words.
5. Return at most {top_k} items.
6. Output ONLY JSON array of strings. Example: ["fort bragg", "bds movement"]

Post:
\"\"\"{text}\"\"\"
Return ONLY JSON:
"""

def call_ollama_for_keywords(
    text: str,
    model: str,
    top_k: int = DEFAULT_TOP_K,
    timeout: int = 40,
) -> List[str]:
    prompt = PROMPT_TEMPLATE.format(text=text, top_k=top_k)

    try:
        resp = requests.post(
            OLLAMA_ENDPOINT,
            json={"model": model, "prompt": prompt, "stream": True},
            timeout=timeout,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Ollama request failed: {e}")
        return []

    chunks = []
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("done"):
            break
        part = obj.get("response", "")
        if part:
            chunks.append(part)

    full_text = "".join(chunks).strip()

    # try direct JSON
    try:
        parsed = json.loads(full_text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()][:top_k]
    except json.JSONDecodeError:
        # maybe JSON inside
        match = re.search(r"\[.*\]", full_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()][:top_k]
            except json.JSONDecodeError:
                pass

    print(f"[WARN] couldn't parse Ollama output: {full_text[:200]}...")
    return []

# =========================================================
# STRICT FILTER: keep only keywords that appear in text
# =========================================================
def filter_by_presence(keywords: List[str], original_cleaned: str) -> List[str]:
    """
    Keep only kws where the normalized kw is a substring of the normalized tweet.
    This is what makes hallucination impossible.
    """
    base = norm(original_cleaned)
    kept = []
    for kw in keywords:
        kw_n = norm(kw)
        if kw_n and kw_n in base:
            kept.append(kw)
    return kept

# =========================================================
# PROCESSING
# =========================================================
def process_folder_with_ollama_strict(
    folder: str,
    output_csv: str,
    screen_names: Iterable[str],
    model: str = DEFAULT_OLLAMA_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    top_k: int = DEFAULT_TOP_K,
    sleep_between: float = 0.0,
) -> None:
    targets: Set[str] = {s.lower().strip() for s in screen_names}
    rows_buffer: List[Dict[str, Any]] = []
    wrote_header = False

    for tweet in iter_tweets_from_folder(folder):
        user = tweet.get("user") or {}
        sn = (user.get("screen_name") or "").lower().strip()
        if targets and sn not in targets:
            continue

        raw_msg = tweet.get("full_text") or tweet.get("text") or ""
        if is_retweet(tweet, raw_msg):
            continue

        rows_buffer.append({
            "screen_name": sn,
            "message": raw_msg,
            "created_at": tweet.get("created_at", ""),
            "tweet_id": tweet.get("id_str") or tweet.get("id") or "",
        })

        if len(rows_buffer) >= batch_size:
            _process_batch_ollama_strict(
                rows_buffer, output_csv, wrote_header, model, top_k, sleep_between
            )
            wrote_header = True
            rows_buffer = []

    if rows_buffer:
        _process_batch_ollama_strict(
            rows_buffer, output_csv, wrote_header, model, top_k, sleep_between
        )

def _process_batch_ollama_strict(
    rows: List[Dict[str, Any]],
    output_csv: str,
    wrote_header: bool,
    model: str,
    top_k: int,
    sleep_between: float,
) -> None:
    print(f"[INFO] processing batch of {len(rows)} (strict) with Ollama={model} ...")
    df = pd.DataFrame(rows)

    all_keywords: List[List[str]] = []
    for msg in df["message"].tolist():
        cleaned = light_clean(msg)
        kws = call_ollama_for_keywords(cleaned, model=model, top_k=top_k)
        kws = filter_by_presence(kws, cleaned)  # <-- the guardrail
        all_keywords.append(kws)
        if sleep_between > 0:
            time.sleep(sleep_between)

    max_k = max((len(k) for k in all_keywords), default=0)
    for i in range(max_k):
        df[f"ollama_kw{i+1}"] = [k[i] if i < len(k) else "" for k in all_keywords]

    # drop rows with no keywords (optional, but usually nicer)
    if max_k > 0:
        kw_cols = [f"ollama_kw{i+1}" for i in range(max_k)]
        df = df[df[kw_cols].notna().any(axis=1)]

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

    parser = argparse.ArgumentParser("STRICT Ollama keyword extractor (no hallucinations)")
    parser.add_argument("--folder", help="folder with .json.gz files")
    parser.add_argument("--out", help="output CSV path")
    parser.add_argument("--users", nargs="+", help="screen names to keep (no @)")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--sleep", type=float, default=0.0)
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
    print(f"[INFO] model: {args.model}")

    process_folder_with_ollama_strict(
        folder=folder,
        output_csv=out_path,
        screen_names=users,
        model=args.model,
        batch_size=args.batch,
        top_k=args.topk,
        sleep_between=args.sleep,
    )

if __name__ == "__main__":
    main()
