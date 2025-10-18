# tweet_monitor_hourly.py

import asyncio
import inspect
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import schedule
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from tweety import TwitterAsync  # type: ignore

# ----------------------------
# CONFIGURATION
# ----------------------------
accounts = ["elonmusk", "CathieDWood"]
min_engagement = 1000              # Minimum likes + retweets + replies for market-moving
market_threshold = 0.8            # Probability threshold for market relevance
db_file = "tweets.db"
tweet_limit = 2                 # Number of tweets to fetch per account per run

load_dotenv()

# ----------------------------
# INITIALIZE AI MODELS
# ----------------------------
market_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)

sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")


# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def _attr_path(obj, *paths, default=None):
    for path in paths:
        if obj is None:
            break
        current = obj
        for part in path.split("."):
            current = getattr(current, part, None)
            if current is None:
                break
        else:
            if current is not None:
                return current
    return default


async def _attempt_method_call(method: Any, attempts: Iterable[Tuple[Tuple[Any, ...], Dict[str, Any]]]) -> bool:
    for args, kwargs in attempts:
        try:
            result = method(*args, **(kwargs.copy() if kwargs else {}))
        except TypeError:
            continue
        if inspect.isawaitable(result):
            await result
        return True
    return False


async def _apply_auth_token(client: TwitterAsync, token: str) -> bool:
    attempts = [
        ((), {"auth_token": token}),
        ((), {"token": token}),
        ((token,), {}),
    ]
    for method_name in ("load_auth_token", "set_auth_token", "authorize"):
        method = getattr(client, method_name, None)
        if method and await _attempt_method_call(method, attempts):
            return True
    return False


async def _apply_cookies(client: TwitterAsync, cookie_path: Path) -> bool:
    path_str = str(cookie_path)
    attempts = [
        ((path_str,), {}),
        ((), {"file_path": path_str}),
        ((), {"path": path_str}),
    ]
    for method_name in ("load_cookies", "load_cookie_file"):
        method = getattr(client, method_name, None)
        if method and await _attempt_method_call(method, attempts):
            return True
    return False


async def _login_tweety_client() -> TwitterAsync:
    session_name = os.getenv("TWITTER_SESSION_NAME", "tweet-monitor-session")
    auth_token = os.getenv("TWITTER_AUTH_TOKEN")
    cookies_path = os.getenv("TWITTER_COOKIES_PATH")
    email = os.getenv("TWITTER_EMAIL")
    login_username = os.getenv("TWITTER_USERNAME") or os.getenv("TWITTER_LOGIN")
    password = os.getenv("TWITTER_PASSWORD")

    client = TwitterAsync(session_name=session_name)
    using_token = bool(auth_token or cookies_path)

    try:
        if using_token:
            if auth_token:
                token_applied = await _apply_auth_token(client, auth_token)
                if not token_applied:
                    raise RuntimeError("Tweety client does not support auth token login.")
            if cookies_path:
                cookie_file = Path(cookies_path).expanduser()
                if not cookie_file.exists():
                    raise FileNotFoundError(f"Cookie file not found: {cookie_file}")
                cookies_applied = await _apply_cookies(client, cookie_file)
                if not cookies_applied:
                    raise RuntimeError("Tweety client cannot load cookies from the provided path.")
            return client

        if not password or not (email or login_username):
            raise RuntimeError(
                "Missing Tweety credentials. Set TWITTER_EMAIL (or TWITTER_USERNAME) "
                "and TWITTER_PASSWORD in your environment."
            )

        login_kwargs: Dict[str, Any] = {"password": password}
        if login_username:
            login_kwargs["username"] = login_username
        if email:
            login_kwargs.setdefault("email", email)

        attempts: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = [((), login_kwargs)]
        if login_username:
            attempts.append(((login_username, password), {}))
        elif email:
            attempts.append(((email, password), {}))

        for method_name in ("sign_in", "login", "signin"):
            method = getattr(client, method_name, None)
            if method and await _attempt_method_call(method, attempts):
                return client

        raise AttributeError("Tweety client has no async login method.")
    except Exception as exc:
        await _maybe_close_client(client)
        message = (
            f"Failed to initialize tweety-ns via auth token/cookies: {exc}"
            if using_token
            else f"Failed to authenticate with tweety-ns: {exc}"
        )
        raise RuntimeError(message) from exc


async def _maybe_close_client(client: Any) -> None:
    close_method = getattr(client, "close", None)
    if close_method is None:
        return
    try:
        result = close_method()
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        pass  # best effort; avoid masking upstream errors


def _existing_tweet_ids(table_name: str = "tweets") -> Set[str]:
    if not Path(db_file).exists():
        return set()

    conn = sqlite3.connect(db_file)
    try:
        cur = conn.execute(
            f"SELECT tweet_id FROM {table_name} WHERE tweet_id IS NOT NULL"
        )
        return {str(row[0]) for row in cur.fetchall() if row[0]}
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()


async def _fetch_tweets_async(
    target_username: str,
    limit: int,
    *,
    store: bool,
    verbose: bool,
) -> pd.DataFrame:
    client = await _login_tweety_client()

    try:
        if hasattr(client, "get_user_tweets"):
            tweets_iterable = await client.get_user_tweets(username=target_username, pages=1)
        elif hasattr(client, "get_tweets"):
            tweets_iterable = await client.get_tweets(username=target_username, pages=1)
        else:
            raise AttributeError("Tweety client does not expose a tweet retrieval method.")

        tweets = list(tweets_iterable or [])
        items: List[Dict[str, Any]] = []
        raw_tweets: List[Any] = []

        for tweet in tweets[:limit]:
            tweet_id = _attr_path(tweet, "id", "tweet_id")
            created_at = _attr_path(tweet, "created_on", "date", "created_at")
            text = (
                _attr_path(tweet, "text", "full_text", "tweet_text", "rawContent", "content") or ""
            )

            likes = _attr_path(
                tweet,
                "stats.likes",
                "stats.likes_count",
                "likes",
                "favorite_count",
            ) or 0
            retweets = _attr_path(
                tweet,
                "stats.retweets",
                "stats.retweets_count",
                "retweets",
                "retweet_count",
            ) or 0
            replies = _attr_path(
                tweet,
                "stats.replies",
                "stats.replies_count",
                "replies",
                "reply_count",
            ) or 0

            items.append(
                {
                    "tweet_id": str(tweet_id) if tweet_id is not None else None,
                    "username": target_username,
                    "created_at": created_at,
                    "text": text,
                    "likes": int(likes),
                    "retweets": int(retweets),
                    "replies": int(replies),
                }
            )
            raw_tweets.append(tweet)

        if store:
            existing_ids = _existing_tweet_ids()
            if existing_ids:
                before = len(items)
                keep_indices: List[int] = []
                filtered_items: List[Dict[str, Any]] = []

                for idx, item in enumerate(items):
                    tweet_id_value = item.get("tweet_id")
                    if tweet_id_value and tweet_id_value in existing_ids:
                        continue
                    filtered_items.append(item)
                    keep_indices.append(idx)

                skipped = before - len(filtered_items)
                if skipped and verbose:
                    print(f"Skipped {skipped} existing tweets for @{target_username}.")

                items = filtered_items
                raw_tweets = [raw_tweets[idx] for idx in keep_indices]

        df = pd.DataFrame(items)
        if df.empty:
            print(f"No new tweets for @{target_username}.")
            print(f"Fetched 0 tweets from @{target_username} via tweety-ns")
            return df

        df, missing_stats = _enrich_tweet_dataframe(df)

        zero_mask = (df["likes"] == 0) & (df["retweets"] == 0) & (df["replies"] == 0)
        if zero_mask.any():
            zero_ids = df.loc[zero_mask, "tweet_id"].tolist()
            warning_parts = [
                "Warning: engagement metrics defaulted to 0.",
                f"Tweets: {', '.join(filter(None, zero_ids)) or 'unknown ids'}.",
            ]
            if missing_stats:
                detail = "; ".join(
                    f"{tid or 'unknown'} missing {', '.join(fields)}"
                    for tid, fields in missing_stats
                )
                warning_parts.append(f"Missing raw stats for: {detail}.")
            print(" ".join(warning_parts))

            if verbose:
                print("Raw tweet payloads with zeroed engagement:")
                for zero_id in zero_ids:
                    for raw in raw_tweets:
                        raw_id = _attr_path(raw, "id", "tweet_id")
                        if str(raw_id) == str(zero_id):
                            print(f"--- tweet_id={zero_id} ---")
                            print(raw)
                            break

        if store:
            store_in_sqlite(df, table_name="tweets")

        if verbose and not df.empty:
            display_cols = [
                col
                for col in [
                    "tweet_id",
                    "created_at",
                    "cleaned",
                    "sentiment",
                    "engagement",
                    "market_related_prob",
                ]
                if col in df.columns
            ]
            print(f"\n--- Tweets fetched for @{target_username} ---")
            print(df[display_cols])

        print(f"Fetched {len(df)} tweets from @{target_username} via tweety-ns")
        return df
    except Exception as exc:
        print(f"Error fetching tweets for {target_username} via tweety-ns: {exc}")
        return pd.DataFrame()
    finally:
        await _maybe_close_client(client)


def _enrich_tweet_dataframe(
    df: pd.DataFrame,
    *,
    threshold: Optional[float] = None,
    engagement_floor: Optional[int] = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    enriched = df.copy()

    if "text" not in enriched.columns and "content" in enriched.columns:
        enriched["text"] = enriched["content"]

    missing_stats: List[Tuple[Optional[str], List[str]]] = []
    for idx, row in enriched.iterrows():
        missing_fields = []
        normalized_values: Dict[str, int] = {}

        for col in ("likes", "retweets", "replies"):
            value = row.get(col)
            try:
                normalized_values[col] = int(value)
            except (TypeError, ValueError):
                normalized_values[col] = 0
                missing_fields.append(col)

        for col, norm_val in normalized_values.items():
            enriched.at[idx, col] = norm_val

        if missing_fields:
            missing_stats.append((row.get("tweet_id"), missing_fields))

    enriched["cleaned"] = enriched["text"].apply(clean_text)
    enriched["market_related_prob"] = enriched["cleaned"].apply(classify_market_related)
    enriched["sentiment"] = enriched["cleaned"].apply(analyze_sentiment_finbert)
    enriched["engagement"] = (
        enriched["likes"] + enriched["retweets"] + enriched["replies"]
    ).astype(int)

    threshold = market_threshold if threshold is None else threshold
    engagement_floor = min_engagement if engagement_floor is None else engagement_floor
    enriched["market_moving"] = (
        (enriched["market_related_prob"] > float(threshold))
        & (enriched["engagement"] > int(engagement_floor))
    ).astype(bool)

    return enriched, missing_stats


def fetch_tweets(username, limit=tweet_limit, *, store: bool = True, verbose: bool = False):
    """
    Fetch recent tweets using tweety-ns to avoid Twitter API rate limits.
    """
    return asyncio.run(_fetch_tweets_async(username, limit, store=store, verbose=verbose))


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    return text


def classify_market_related(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    labels = ["market related", "not market related"]
    try:
        result = market_classifier(text, candidate_labels=labels)
        scores_dict = dict(zip(result["labels"], result["scores"]))
        return scores_dict.get("market related", 0.0)
    except ValueError:
        return 0.0


def analyze_sentiment_finbert(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = sentiment_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    pos, neu, neg = probs[0]
    return float(pos - neg)  # +1 = positive, -1 = negative, 0 = neutral


def store_in_sqlite(df, table_name="tweets"):
    if df.empty:
        print(f"No data to store for table '{table_name}'.")
        return

    if "tweet_id" in df.columns:
        df = df.drop_duplicates(subset="tweet_id", keep="first")

    conn = sqlite3.connect(db_file)
    try:
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone() is not None

        if table_exists:
            cur = conn.execute(f"PRAGMA table_info({table_name})")
            existing_cols = {row[1] for row in cur.fetchall()}

            if "tweet_id" not in existing_cols:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN tweet_id TEXT")
                conn.commit()

            if "tweet_id" in df.columns:
                existing_ids = {
                    row[0]
                    for row in conn.execute(
                        f"SELECT tweet_id FROM {table_name} WHERE tweet_id IS NOT NULL"
                    )
                }
                if existing_ids:
                    df = df[~df["tweet_id"].isin(existing_ids)]

        if df.empty:
            print(f"No new rows to insert into '{table_name}'.")
            return

        df.to_sql(table_name, conn, if_exists="append", index=False)

        if "tweet_id" in df.columns:
            conn.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_tweet_id "
                f"ON {table_name}(tweet_id)"
            )
            conn.commit()
    finally:
        conn.close()


# ----------------------------
# MONITOR FUNCTION
# ----------------------------
def monitor():
    all_tweets = pd.DataFrame()

    for account in accounts:
        df = fetch_tweets(account)
        if df.empty:
            continue
        all_tweets = pd.concat([all_tweets, df], ignore_index=True)

    if all_tweets.empty:
        print("No tweets gathered during this cycle.")
        return

    if "market_moving" in all_tweets.columns:
        market_mask = all_tweets["market_moving"].astype(bool)
    else:
        market_mask = pd.Series([False] * len(all_tweets), index=all_tweets.index)

    market_tweets = all_tweets[market_mask]
    if not market_tweets.empty:
        print(f"\n--- Market-moving tweets detected at {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
        print(market_tweets[["created_at", "cleaned", "sentiment", "engagement", "market_related_prob"]])
        store_in_sqlite(market_tweets, table_name="market_moving_tweets")


def evaluate_tweets_from_db(
    db_file: str,
    table_name: str = "tweets",
    market_threshold: float = market_threshold,
    min_engagement: int = min_engagement,
) -> pd.DataFrame:
    """
    Evaluate tweets and persist computed fields into SQLite if they don't exist.
    """
    computed_cols: Dict[str, str] = {
        "cleaned": "TEXT",
        "market_related_prob": "REAL",
        "sentiment": "REAL",
        "engagement": "INTEGER",
        "market_moving": "INTEGER",
    }

    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(f"PRAGMA table_info({table_name})")
        cols_info = cur.fetchall()
        if not cols_info:
            raise ValueError(f"Table '{table_name}' not found in database '{db_file}'.")

        existing_cols = {row["name"] for row in cols_info}
        pk_cols = [row["name"] for row in cols_info if row["pk"] == 1]
        pk_col = pk_cols[0] if pk_cols else None

        for col, col_type in computed_cols.items():
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {col_type}")
        conn.commit()

        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    finally:
        conn.close()


# ----------------------------
# SCHEDULE MONITORING
# ----------------------------
schedule.every(1).hours.do(monitor)  # Run hourly

print("Hourly monitoring started. Call fetch_tweets('@handle', limit=10, verbose=True) for debugging.\n")

# while True:
#     schedule.run_pending()
#     time.sleep(60)
