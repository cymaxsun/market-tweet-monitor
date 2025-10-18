from __future__ import annotations
import html
import sqlite3
import textwrap
from pathlib import Path
from numbers import Integral
from typing import Optional, Tuple

import pandas as pd
import streamlit as st


DB_FILE = Path(__file__).resolve().parent.parent / "tweets.db"


st.set_page_config(page_title="Tweet Market Monitor", layout="wide")
st.title("📈 Market Tweet Monitor Dashboard")

TWEET_CARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
.tweet-card-container {
    display: flex;
    justify-content: center;
    background: transparent;
    margin: 0 auto;
    width: 100%;
}
.tweet-card {
    background: linear-gradient(160deg, rgba(21, 24, 38, 0.98), rgba(25, 19, 51, 0.95));
    border-radius: 18px;
    padding: 1.85rem;
    border: 1px solid rgba(120, 132, 255, 0.35);
    box-shadow: 0 18px 38px rgba(25, 18, 51, 0.55);
    color: #f1f4ff;
    font-family: "Poppins", "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    position: relative;
    overflow: hidden;
    max-width: 720px;
    width: 100%;
}
.tweet-card::before,
.tweet-card::after {
    content: "";
    position: absolute;
    border-radius: 50%;
    filter: blur(0.6px);
}
.tweet-card::before {
    top: -30%;
    right: -25%;
    width: 260px;
    height: 260px;
    background: radial-gradient(circle at center, rgba(255, 119, 57, 0.35), transparent 70%);
}
.tweet-card::after {
    bottom: -40%;
    left: -20%;
    width: 240px;
    height: 240px;
    background: radial-gradient(circle at center, rgba(74, 161, 255, 0.4), transparent 75%);
}
.tweet-card::selection {
    background: rgba(255, 150, 102, 0.35);
}
.tweet-card-content {
    position: relative;
    z-index: 1;
    width: 100%;
}
.tweet-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 1rem;
    font-size: 0.95rem;
    font-weight: 600;
    color: #d7dcff;
}
.tweet-header span:first-child {
    color: #ffb48a;
}
.tweet-body {
    font-size: 1.12rem;
    line-height: 1.7;
    margin-bottom: 1.2rem;
    white-space: pre-wrap;
    color: #fdf7ff;
    text-shadow: 0 1px 12px rgba(0, 0, 0, 0.25);
}
.tweet-metrics {
    display: flex;
    gap: 0.75rem;
    font-size: 0.9rem;
    flex-wrap: wrap;
    color: #fcefff;
}
.tweet-metrics span {
    color: inherit;
    background: linear-gradient(135deg, rgba(255, 133, 82, 0.32), rgba(97, 143, 255, 0.25));
    border-radius: 12px;
    padding: 0.45rem 0.78rem;
    border: 1px solid rgba(255, 183, 128, 0.35);
    box-shadow: inset 0 0 0 rgba(255, 255, 255, 0.15), 0 4px 14px rgba(27, 32, 55, 0.35);
    backdrop-filter: blur(6px);
}
.tweet-metrics span strong {
    margin-right: 0.4rem;
    color: #ffe4d4;
}
.tweet-nav {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
}
.tweet-nav button {
    background: linear-gradient(135deg, rgba(255, 122, 87, 0.78), rgba(99, 145, 255, 0.75));
    border: 1px solid rgba(255, 154, 110, 0.7);
    color: #fef8ff;
    font-weight: 600;
    letter-spacing: 0.04em;
    border-radius: 14px;
    padding: 0;
    box-shadow: 0 16px 28px rgba(36, 28, 63, 0.4);
    min-width: 3.4rem;
    max-width: 3.4rem;
    min-height: 3rem;
    height: auto;
    font-size: 1.45rem;
    display: flex;
    align-items: center;
    justify-content: center;
}
.tweet-nav button:disabled {
    background: rgba(37, 42, 63, 0.75);
    border-color: rgba(81, 92, 118, 0.5);
    color: rgba(211, 217, 238, 0.55);
    box-shadow: none;
}
</style>
"""


@st.cache_data(show_spinner=False)
def load_tweets(db_path: Path) -> pd.DataFrame:
    """Load tweets from SQLite, caching the result to keep the UI responsive."""
    if not db_path.exists():
        return pd.DataFrame()

    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query("SELECT * FROM tweets", conn)
    except sqlite3.Error as exc:
        # Bubble up an empty frame; the caller will surface a message.
        st.error(f"Database error: {exc}")
        return pd.DataFrame()


def apply_filters(df: pd.DataFrame, username: Optional[str], date_range: Optional[Tuple]) -> pd.DataFrame:
    """Apply sidebar filters without mutating the cached dataframe."""
    if df.empty:
        return df

    filtered = df.copy()

    if username:
        mask = filtered["username"].str.contains(username, case=False, na=False)
        filtered = filtered[mask]

    if date_range and "created_at" in filtered.columns:
        start_date, end_date = date_range
        if start_date and end_date:
            start_ts = pd.to_datetime(start_date)
            end_ts = pd.to_datetime(end_date)
            if start_ts > end_ts:
                start_ts, end_ts = end_ts, start_ts

            created_at = pd.to_datetime(filtered["created_at"], errors="coerce")
            filtered = filtered[
                (created_at >= start_ts)
                & (created_at <= end_ts)
            ]

    return filtered


def render_summary(df: pd.DataFrame) -> None:
    """Display headline metrics so users can orient quickly."""
    total = int(len(df))
    movers = int(df.get("market_moving", pd.Series(dtype=int)).astype(int).sum()) if total else 0
    avg_sentiment = float(df.get("sentiment", pd.Series(dtype=float)).mean()) if total else 0.0

    col_total, col_movers, col_sentiment = st.columns(3)
    col_total.metric("Tweets", f"{total:,}")
    col_movers.metric("Market-moving", f"{movers:,}")
    col_sentiment.metric("Avg. sentiment", f"{avg_sentiment:.2f}")


def _format_metric(value, suffix: str = "") -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    if isinstance(value, Integral):
        return f"{value:,}{suffix}"
    return f"{value}{suffix}"


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return pd.isna(value)
    except TypeError:
        return False


def render_tweet_carousel(df: pd.DataFrame) -> None:
    """Display tweets in a carousel-style layout with navigation buttons."""
    if df.empty:
        st.info("No tweets to display for the current filters.")
        return

    display_df = df.copy()
    if "created_at" in display_df.columns:
        display_df = display_df.sort_values(by="created_at", ascending=False)
    display_df = display_df.reset_index(drop=True)

    total = len(display_df)
    if "carousel_index" not in st.session_state:
        st.session_state.carousel_index = 0

    st.session_state.carousel_index = max(0, min(st.session_state.carousel_index, total - 1))

    current_idx = st.session_state.carousel_index
    tweet = display_df.iloc[current_idx]

    st.markdown(TWEET_CARD_CSS, unsafe_allow_html=True)

    st.caption(f"Tweet {current_idx + 1} of {total}")
    try:
        nav_prev, nav_card, nav_next = st.columns([0.7, 6, 0.7], vertical_alignment="center")
    except TypeError:
        nav_prev, nav_card, nav_next = st.columns([0.7, 6, 0.7])

    with nav_prev:
        st.markdown('<div class="tweet-nav">', unsafe_allow_html=True)
        if st.button("◀", key="tweet_prev", use_container_width=True, disabled=current_idx == 0):
            st.session_state.carousel_index -= 1
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with nav_next:
        st.markdown('<div class="tweet-nav">', unsafe_allow_html=True)
        if st.button("▶", key="tweet_next", use_container_width=True, disabled=current_idx == total - 1):
            st.session_state.carousel_index += 1
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    created_at = tweet.get("created_at")
    created_display = pd.to_datetime(created_at).strftime("%Y-%m-%d %H:%M") if pd.notna(created_at) else "Unknown"
    raw_text = tweet.get("cleaned")
    if _is_empty(raw_text):
        raw_text = tweet.get("text")
    if _is_empty(raw_text):
        cleaned_text_html = "No text available."
    else:
        cleaned_text_html = html.escape(str(raw_text)).replace("\n", "<br>")

    username = tweet.get("username", "unknown")
    username_html = html.escape(str(username))

    card_html = textwrap.dedent(
        f"""
    <div class="tweet-card-container">
        <div class="tweet-card">
            <div class="tweet-card-content">
                <div class="tweet-header">
                    <span>@{username_html}</span>
                    <span>{created_display}</span>
                </div>
                <div class="tweet-body">{cleaned_text_html}</div>
                <div class="tweet-metrics">
                    <span><strong>Sentiment</strong>{_format_metric(tweet.get("sentiment"))}</span>
                    <span><strong>Engagement</strong>{_format_metric(tweet.get("engagement"))}</span>
                    <span><strong>Likes</strong>{_format_metric(tweet.get("likes"))}</span>
                    <span><strong>Retweets</strong>{_format_metric(tweet.get("retweets"))}</span>
                    <span><strong>Replies</strong>{_format_metric(tweet.get("replies"))}</span>
                </div>
            </div>
        </div>
    </div>
    """
    )

    with nav_card:
        st.markdown(card_html, unsafe_allow_html=True)


def main() -> None:
    st.sidebar.header("Filters")
    username = st.sidebar.text_input("Username contains")

    # Optional date filtering when created_at is present.
    date_range: Optional[tuple] = None
    date_filter_enabled = st.sidebar.checkbox("Filter by date range")
    if date_filter_enabled:
        default_start = pd.Timestamp.now() - pd.Timedelta(days=7)
        default_end = pd.Timestamp.now()
        start = st.sidebar.date_input("Start date", default_start)
        end = st.sidebar.date_input("End date", default_end)
        date_range = (start, end)

    df = load_tweets(DB_FILE)
    if df.empty:
        st.info("No tweets available yet. Check back after the monitor ingests new data.")
        return

    filtered_df = apply_filters(df, username, date_range)
    st.subheader(f"Showing {len(filtered_df)} tweets")

    if filtered_df.empty:
        st.warning("No tweets match the selected filters. Adjust them to see data.")
        return

    render_summary(filtered_df)
    render_tweet_carousel(filtered_df)


if __name__ == "__main__":
    main()
