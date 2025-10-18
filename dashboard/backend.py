from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import sqlite3
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from tweet_monitor import fetch_tweets as fetch_latest_tweets  # type: ignore
except ImportError:  # pragma: no cover
    fetch_latest_tweets = None  # type: ignore


DB_PATH = Path(__file__).resolve().parent.parent / "tweets.db"

app = FastAPI(title="Tweet Monitor API")

# Allow local development frontends (Vite dev server, CRA, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_tweets_frame() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise HTTPException(status_code=500, detail="tweets.db not found.")

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM tweets", conn)

    if df.empty:
        return df

    # Normalize types
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    numeric_cols = ["sentiment", "market_related_prob", "engagement", "likes", "retweets", "replies"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def apply_filters(
    df: pd.DataFrame,
    username: Optional[str],
    start: Optional[datetime],
    end: Optional[datetime],
) -> pd.DataFrame:
    filtered = df.copy()

    if username:
        filtered = filtered[filtered["username"].str.contains(username, case=False, na=False)]

    if start:
        filtered = filtered[filtered["created_at"] >= start]
    if end:
        filtered = filtered[filtered["created_at"] <= end]

    return filtered


def make_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"count": 0, "market_moving": 0, "avg_sentiment": 0.0}

    count = int(len(df))
    market = int(df.get("market_moving", pd.Series(dtype=int)).fillna(0).astype(int).sum())
    avg_sentiment = float(df.get("sentiment", pd.Series(dtype=float)).mean() or 0.0)

    return {
        "count": count,
        "market_moving": market,
        "avg_sentiment": round(avg_sentiment, 3),
    }


def serialize_tweets(df: pd.DataFrame) -> List[dict]:
    records: List[dict] = []

    for row in df.to_dict(orient="records"):
        created_at = row.get("created_at")
        if isinstance(created_at, pd.Timestamp):
            created_iso = created_at.isoformat()
        elif isinstance(created_at, datetime):
            created_iso = created_at.isoformat()
        else:
            created_iso = None

        records.append(
            {
                "tweet_id": row.get("tweet_id"),
                "username": row.get("username"),
                "created_at": created_iso,
                "text": row.get("text") or "",
                "cleaned": row.get("cleaned") or row.get("text") or "",
                "sentiment": float(row.get("sentiment")) if row.get("sentiment") is not None else None,
                "engagement": int(row.get("engagement")) if pd.notna(row.get("engagement")) else None,
                "likes": int(row.get("likes")) if pd.notna(row.get("likes")) else None,
                "retweets": int(row.get("retweets")) if pd.notna(row.get("retweets")) else None,
                "replies": int(row.get("replies")) if pd.notna(row.get("replies")) else None,
                "market_related_prob": float(row.get("market_related_prob"))
                if row.get("market_related_prob") is not None
                else None,
                "market_moving": bool(row.get("market_moving")) if row.get("market_moving") is not None else False,
            }
        )

    return records


DEFAULT_LIVE_LIMIT = 4


class FetchTweetsRequest(BaseModel):
    username: str = Field(..., description="Twitter handle without @")
    limit: int = Field(DEFAULT_LIVE_LIMIT, ge=1, le=50, description="Maximum tweets to fetch")
    store: bool = Field(True, description="Persist fetched tweets to the database")


def _normalise_username(raw: str) -> str:
    username = (raw or "").strip()
    while username.startswith("@"):
        username = username[1:]
    return username


@app.get("/api/tweets")
def read_tweets(
    username: Optional[str] = Query(default=None, description="Case-insensitive partial username filter."),
    start_date: Optional[str] = Query(default=None, description="ISO date string (YYYY-MM-DD)."),
    end_date: Optional[str] = Query(default=None, description="ISO date string (YYYY-MM-DD)."),
) -> dict:
    df = load_tweets_frame()

    start_dt = datetime.fromisoformat(start_date) if start_date else None
    end_dt = datetime.fromisoformat(end_date) if end_date else None

    filtered = apply_filters(df, username=username, start=start_dt, end=end_dt)
    filtered = filtered.sort_values(by="created_at", ascending=False)

    summary = make_summary(filtered)
    tweets = serialize_tweets(filtered.reset_index(drop=True))

    return {"tweets": tweets, "summary": summary}


@app.post("/api/tweets/fetch")
def fetch_tweets_for_handle(payload: FetchTweetsRequest) -> dict:
    if fetch_latest_tweets is None:  # pragma: no cover
        raise HTTPException(status_code=503, detail="Live tweet fetching is unavailable on this server.")

    username = _normalise_username(payload.username)
    if not username:
        raise HTTPException(status_code=422, detail="A username is required.")

    try:
        df = fetch_latest_tweets(username=username, limit=payload.limit, store=payload.store, verbose=False)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to fetch tweets for @{username}: {exc}") from exc

    if df.empty:
        # Fallback to any cached tweets in the database.
        cache = load_tweets_frame()
        if not cache.empty:
            df = cache[cache["username"].str.lower() == username.lower()].copy()
        else:
            df = pd.DataFrame()

    if not df.empty and "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df = df.sort_values(by="created_at", ascending=False)
        df = df.head(payload.limit)

    tweets = serialize_tweets(df.reset_index(drop=True)) if not df.empty else []
    return {
        "username": username,
        "tweets": tweets,
        "count": len(tweets),
    }
