# Market Tweet Monitor

A full-stack tool for monitoring market-moving tweets. It scrapes targeted Twitter accounts, classifies the impact with transformer models, stores results in SQLite, and exposes them through a FastAPI + React dashboard.

## Features

- **Automated scraping**: Fetch recent tweets using Tweety with cookie/`auth_token` support to avoid API limits.
- **NLP scoring**: Zero-shot classification for market relevance and RoBERTa sentiment analysis.
- **SQLite persistence**: Deduplicate by `tweet_id`, compute metrics (engagement, market_moving) before storage.
- **FastAPI backend**: Filter tweets by handle/date and trigger on-demand fetches.
- **React dashboard**: Monitoring view with configurable watch lists plus an on-demand fetch workspace.

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

## Workflow Notes

- `tweet_monitor.py` handles scheduled scraping and NLP enrichment. Run it directly if you want continuous monitoring.
- `dashboard/backend.py` exposes the REST API and a live fetch endpoint that the UI uses for on-demand requests.
- The React dashboard has two pages: **Monitoring Dashboard** (watch list management, ranked feed, configurable sort/refresh) and **On-Demand Fetch** (single-handle lookup with sentiment snapshot).
- `.gitignore` omits Tweety session files, the SQLite database, and other local artifacts.
- Tweak `MAX_FEED_TWEETS` (frontend) or `DEFAULT_LIVE_LIMIT` (backend) to change how many tweets are surfaced by default.

## License

MIT
