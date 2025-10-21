# tweet_monitor_hourly.py

import asyncio
import inspect
import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import schedule
import numpy as np
import tweetnlp
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, pipeline
from tweety import TwitterAsync  # type: ignore

# ----------------------------
# CONFIGURATION
# ----------------------------
DEFAULT_ACCOUNTS = ["elonmusk", "CathieDWood", "CNBC"]
min_engagement = 1000              # Minimum likes + retweets + replies for market-moving
market_threshold = 0.8            # Probability threshold for market relevance
db_file = "tweets.db"
tweet_limit = 2                 # Number of tweets to fetch per account per run

OPTIONAL_TWEET_COLUMNS = {
    "is_retweet": "INTEGER",
    "retweeted_user": "TEXT",
}

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("tweet-monitor")


def _normalise_handle(value: Optional[str]) -> str:
    username = (value or "").strip()
    while username.startswith("@"):
        username = username[1:]
    return username


def _ensure_accounts_table(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS accounts (username TEXT PRIMARY KEY)")


def load_accounts_from_db(default_accounts: Optional[List[str]] = None) -> List[str]:
    default_accounts = default_accounts or []
    cleaned_defaults = []
    seen = set()
    for handle in default_accounts:
        normalised = _normalise_handle(handle)
        key = normalised.lower()
        if normalised and key not in seen:
            seen.add(key)
            cleaned_defaults.append(normalised)

    if not Path(db_file).exists():
        if cleaned_defaults:
            with sqlite3.connect(db_file) as conn:
                _ensure_accounts_table(conn)
                conn.executemany(
                    "INSERT OR IGNORE INTO accounts(username) VALUES (?)",
                    [(username,) for username in cleaned_defaults],
                )
                conn.commit()
            return cleaned_defaults
        return []

    with sqlite3.connect(db_file) as conn:
        _ensure_accounts_table(conn)
        rows = conn.execute(
            "SELECT username FROM accounts ORDER BY username COLLATE NOCASE"
        ).fetchall()
        seen_accounts: Set[str] = set()
        accounts: List[str] = []
        for row in rows:
            value = row[0] if row else ""
            username = _normalise_handle(value)
            key = username.lower()
            if username and key not in seen_accounts:
                seen_accounts.add(key)
                accounts.append(username)
        if accounts:
            return accounts
    return []

# ----------------------------
# INITIALIZE AI MODELS
# ----------------------------
market_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

model = tweetnlp.Sentiment()

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
        logger.info("Fetching tweets for @%s (limit=%s, store=%s)", target_username, limit, store)
        if hasattr(client, "get_tweets"):
            tweets_iterable = await client.get_tweets(username=target_username, pages=1)
        else:
            raise AttributeError("Tweety client does not expose a tweet retrieval method.")

        tweets = list(tweets_iterable or [])
        logger.debug("Tweety returned %d items for @%s", len(tweets), target_username)
        items: List[Dict[str, Any]] = []
        raw_tweets: List[Any] = []

        for tweet in tweets[:limit]:
            if verbose:
                logger.debug("Processing tweet payload: %s", _attr_path(tweet, "id", "tweet_id"))
            tweet_id = _attr_path(tweet, "id")
            created_at = _attr_path(tweet, "created_on", "date")
            author_username = _attr_path(
                tweet,
                "author.name"
                "author.username",
                "author.screen_name",
            )

            base_text = _attr_path(
                tweet,
                "text",
            )

            is_retweet = _attr_path(
                tweet,
                "is_retweet",
            )


            retweets = _attr_path(
                tweet,
                "retweet_counts",
            ) or 0


            if is_retweet:
                retweeted_text = _attr_path(
                    tweet,
                    "retweeted_tweet.text",
                )
                
                retweeted_user = _attr_path(
                    tweet,
                    "retweeted_tweet.author.name",
                    "retweeted_tweet.author.username",
                    "retweeted_tweet.author.screen_name",
                )
                likes = _attr_path(
                    tweet,
                    "retweeted_tweet.likes",
                ) or 0
                replies = _attr_path(
                    tweet,
                    "retweeted_tweet.reply_counts",
                ) or 0
            else :
                likes = _attr_path(
                    tweet,
                    "likes",
                ) or 0
                replies = _attr_path(
                    tweet,
                    "reply_counts",
                ) or 0
            text = base_text or ""

            

            items.append(
                {
                    "tweet_id": str(tweet_id) if tweet_id is not None else None,
                    "username": target_username,
                    "created_at": created_at,
                    "text": text,
                    "is_retweet": is_retweet,
                    "likes": int(likes),
                    "retweets": int(retweets),
                    "replies": int(replies),
                    "retweeted_user": retweeted_user if is_retweet else None,
                    #"retweeted_text": retweeted_text if is_retweet else None,
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
        logger.debug("Normalised %d rows for @%s", len(df.index), target_username)
        if df.empty:
            print(f"No new tweets for @{target_username}.")
            print(f"Fetched 0 tweets from @{target_username} via tweety-ns")
            return df

        df, missing_stats = _enrich_tweet_dataframe(df)
        if missing_stats:
            logger.debug(
                "Engagement stats missing for %d rows while enriching @%s", len(missing_stats), target_username
            )

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
        logger.info("Persisted %d tweets for @%s", len(df.index), target_username)
        return df
    except Exception as exc:
        logger.exception("Error fetching tweets for @%s via tweety-ns", target_username)
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
        logger.debug("Enrichment skipped: empty DataFrame received")
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

    if missing_stats:
        logger.debug("Normalized engagement counts; pending missing stats for %d rows", len(missing_stats))

    enriched["cleaned"] = enriched["text"].apply(clean_text)
    logger.debug("Applied text cleaning for %d rows", len(enriched.index))

    enriched["market_related_prob"] = enriched["cleaned"].apply(classify_market_related)
    logger.debug("Computed market probabilities")

    sentiment_payloads = enriched["cleaned"].apply(analyze_sentiment)
    def _extract_sentiment_label(payload):
        if not isinstance(payload, dict):
            return "neutral"
        return payload.get("label")
    def _extract_sentiment_probability(payload):
        if not isinstance(payload, dict):
            return 0.0
        label = payload.get("label")
        probabilities = payload.get("probability") or {}
        value = probabilities.get(label)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    enriched["sentiment"] = sentiment_payloads.apply(_extract_sentiment_label)
    enriched["sentiment_prob"] = sentiment_payloads.apply(_extract_sentiment_probability)

    logger.debug(
        "Sentiment enrichment complete: %d labels, %d probabilities",
        enriched["sentiment"].notna().sum(),
        enriched["sentiment_prob"].notna().sum(),
    )

    enriched["engagement"] = (
        enriched["likes"] + enriched["retweets"] + enriched["replies"]
    ).astype(int)

    threshold = market_threshold if threshold is None else threshold
    engagement_floor = min_engagement if engagement_floor is None else engagement_floor
    enriched["market_moving"] = (
        (enriched["market_related_prob"] > float(threshold))
        & (enriched["engagement"] > int(engagement_floor))
    ).astype(bool)
    logger.debug(
        "Flagged %d market-moving tweets (threshold=%s engagement_floor=%s)",
        enriched["market_moving"].sum(),
        threshold,
        engagement_floor,
    )

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
    text = re.sub(r"^RT\s*:?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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


def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return model.sentiment(text, return_probability=True)
    except Exception as error:  # pragma: no cover - model failures are logged upstream
        logger.debug("Sentiment model failure for text '%s': %s", text[:80], error)
        return None


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

            for col_name, col_type in OPTIONAL_TWEET_COLUMNS.items():
                if col_name in df.columns and col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}")
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
            logger.debug("No new rows to insert into '%s'.", table_name)
            return

        
        if "retweeted_user" in df.columns:
            df["retweeted_user"] = (
                df["retweeted_user"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lstrip("@")
                .replace({"": None})
            )

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
    monitored_accounts = load_accounts_from_db(DEFAULT_ACCOUNTS)
    if not monitored_accounts:
        logger.warning("No accounts configured; skipping monitor run.")
        return

    all_tweets = pd.DataFrame()

    for account in monitored_accounts:
        df = fetch_tweets(account)
        if df.empty:
            continue
        all_tweets = pd.concat([all_tweets, df], ignore_index=True)

    if all_tweets.empty:
        logger.info("No tweets gathered during this cycle.")
        return

    if "market_moving" in all_tweets.columns:
        market_mask = all_tweets["market_moving"].astype(bool)
    else:
        market_mask = pd.Series([False] * len(all_tweets), index=all_tweets.index)

    market_tweets = all_tweets[market_mask]
    if not market_tweets.empty:
        logger.info("Market-moving tweets detected (%d rows).", len(market_tweets))
        logger.debug(
            "Sample market movers:\n%s",
            market_tweets[["created_at", "cleaned", "sentiment", "engagement", "market_related_prob"]].head(),
        )


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

        required_cols = {**OPTIONAL_TWEET_COLUMNS, **computed_cols}
        for col, col_type in required_cols.items():
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {col_type}")
        conn.commit()

        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    finally:
        conn.close()
