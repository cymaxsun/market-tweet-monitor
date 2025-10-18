# Market Tweet Monitor

A full-stack tool for monitoring market-moving tweets. It scrapes targeted Twitter accounts, classifies the impact with transformer models, stores results in SQLite, and exposes them through a FastAPI + React dashboard.

## Features

- **Automated scraping**: Fetch recent tweets using Tweety with cookie/`auth_token` support to avoid API limits.
- **NLP scoring**: Zero-shot classification for market relevance and RoBERTa sentiment analysis.
- **SQLite persistence**: Deduplicate by `tweet_id`, compute metrics (engagement, market_moving) before storage.
- **FastAPI backend**: Filter tweets by handle/date and trigger on-demand fetches.
- **React dashboard**: Carousel UI showing the latest 4 tweets per query, with live fetch button.

## Quick Start

1. **Clone & install**
   ```bash
   git clone https://github.com/<your-user>/market-tweet-monitor.git
   cd market-tweet-monitor
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   cd dashboard/react-app && npm install
   ```

2. **Environment**
   Create `.env` with Tweety credentials (choose one login method):
   ```env
   TWITTER_AUTH_TOKEN=...
   # or email/username + TWITTER_PASSWORD
   ```

3. **Run backend**
   ```bash
   uvicorn dashboard.backend:app --reload
   ```

4. **Run frontend**
   ```bash
   cd dashboard/react-app
   npm run dev
   ```
