import { useEffect, useMemo, useState } from "react";
import dayjs from "dayjs";
import TweetCarousel from "./components/TweetCarousel";

const API_URL = "/api/tweets";
const MAX_CAROUSEL_TWEETS = 7;

const initialFilters = {
  username: "",
  startDate: "",
  endDate: "",
  refreshToken: 0,
};

function App() {
  const [filters, setFilters] = useState(() => ({ ...initialFilters }));
  const [pendingUsername, setPendingUsername] = useState(initialFilters.username);
  const [tweets, setTweets] = useState([]);
  const [summary, setSummary] = useState({
    count: 0,
    market_moving: 0,
    avg_sentiment: 0,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [fetchingLatest, setFetchingLatest] = useState(false);
  const [fetchError, setFetchError] = useState(null);
  const [fetchNotice, setFetchNotice] = useState(null);

  useEffect(() => {
    setPendingUsername(filters.username);
  }, [filters.username]);

  useEffect(() => {
    const { username, startDate, endDate } = filters;
    const controller = new AbortController();
    const fetchTweets = async () => {
      setLoading(true);
      setError(null);
      try {
        const params = new URLSearchParams();
        if (username) params.set("username", username);
        if (startDate) params.set("start_date", startDate);
        if (endDate) params.set("end_date", endDate);

        const response = await fetch(
          `${API_URL}?${params.toString()}`,
          { signal: controller.signal },
        );
        if (!response.ok) {
          throw new Error(`API error (${response.status})`);
        }
        const data = await response.json();
        setTweets(data.tweets ?? []);
        setSummary(data.summary ?? { count: 0, market_moving: 0, avg_sentiment: 0 });
      } catch (err) {
        if (err.name !== "AbortError") {
          setError(err.message || "Failed to load tweets.");
        }
      } finally {
        setLoading(false);
      }
    };

    fetchTweets();
    return () => controller.abort();
  }, [filters.username, filters.startDate, filters.endDate, filters.refreshToken]);

  const normaliseHandle = (value) => value.trim().replace(/^@+/, "");

  const commitUsernameFilter = () => {
    const normalized = normaliseHandle(pendingUsername);
    setPendingUsername(normalized);
    setFilters((prev) => ({
      ...prev,
      username: normalized,
      refreshToken: prev.refreshToken + 1,
    }));
  };

  const handleDateChange = (field) => (event) => {
    setFilters((prev) => ({
      ...prev,
      [field]: event.target.value,
    }));
  };

  const handleReset = () => {
    setFilters({ ...initialFilters });
    setPendingUsername(initialFilters.username);
    setFetchError(null);
    setFetchNotice(null);
  };

  const uniqueUsernames = useMemo(() => {
    const set = new Set(tweets.map((tweet) => tweet.username).filter(Boolean));
    return Array.from(set);
  }, [tweets]);

  const carouselTweets = useMemo(
    () => tweets.slice(0, MAX_CAROUSEL_TWEETS),
    [tweets],
  );

  const handleFetchLatest = async () => {
    const handle = normaliseHandle(pendingUsername);
    if (!handle) {
      setFetchError("Enter a handle to pull tweets.");
      setFetchNotice(null);
      return;
    }

    setFetchingLatest(true);
    setFetchError(null);
    setFetchNotice(null);

    try {
      const response = await fetch("/api/tweets/fetch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: handle, limit: MAX_CAROUSEL_TWEETS }),
      });

      if (!response.ok) {
        let detail = `Request failed (${response.status})`;
        try {
          const payload = await response.json();
          if (payload?.detail) detail = payload.detail;
        } catch (parseErr) {
          // ignore JSON parse errors; fall back to default detail
        }
        throw new Error(detail);
      }

      const data = await response.json();
      const count = data?.count ?? (data?.tweets?.length ?? 0);
      setFetchNotice(
        count
          ? `Fetched ${count} tweet${count === 1 ? "" : "s"} for @${handle}.`
          : `No new tweets found for @${handle}.`,
      );
      setPendingUsername(handle);
      setFilters((prev) => ({
        ...prev,
        username: handle,
        refreshToken: prev.refreshToken + 1,
      }));
    } catch (err) {
      setFetchError(err.message || "Unable to fetch tweets.");
    } finally {
      setFetchingLatest(false);
    }
  };

  return (
    <div className="page-shell">
      <header className="header">
        <h1>📈 Market Tweet Monitor</h1>
        <p className="tagline">
          Track high-signal tweets and sentiment from your watchlist.
        </p>
      </header>

      <section className="filters">
        <div className="field">
          <label htmlFor="username">Username</label>
          <input
            id="username"
            type="text"
            placeholder="elonmusk"
            value={pendingUsername}
            onChange={(event) => setPendingUsername(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                commitUsernameFilter();
              }
            }}
            list="username-suggestions"
          />
          <datalist id="username-suggestions">
            {uniqueUsernames.map((name) => (
              <option key={name} value={name} />
            ))}
          </datalist>
        </div>
        <div className="field field-action">
          <label htmlFor="fetch-handle">Pull tweets</label>
          <button
            id="fetch-handle"
            type="button"
            className="fetch-btn"
            onClick={handleFetchLatest}
            disabled={fetchingLatest || !normaliseHandle(pendingUsername)}
          >
            {fetchingLatest ? "Fetching…" : "Fetch latest"}
          </button>
          {(fetchError || fetchNotice) && (
            <p className={`helper-text ${fetchError ? "error" : "success"}`}>
              {fetchError || fetchNotice}
            </p>
          )}
        </div>
        <div className="field">
          <label htmlFor="startDate">Start date</label>
          <input
            id="startDate"
            type="date"
            value={filters.startDate}
            onChange={handleDateChange("startDate")}
          />
        </div>
        <div className="field">
          <label htmlFor="endDate">End date</label>
          <input
            id="endDate"
            type="date"
            value={filters.endDate}
            onChange={handleDateChange("endDate")}
          />
        </div>
        <button className="reset-btn" type="button" onClick={handleReset}>
          Reset
        </button>
      </section>

      <section className="summary-grid">
        <SummaryCard title="Tweets" value={summary.count} />
        <SummaryCard title="Market-moving" value={summary.market_moving} />
        <SummaryCard title="Avg. sentiment" value={summary.avg_sentiment} />
      </section>

      {error && <div className="error-banner">{error}</div>}

      <TweetCarousel tweets={carouselTweets} loading={loading} />
      {tweets.length > carouselTweets.length && (
        <p className="carousel-note">
          Showing the latest {carouselTweets.length} of {tweets.length} tweets. Refine filters for more.
        </p>
      )}

      <footer className="footer">
        <span>
          Data source: SQLite snapshot (
          {dayjs().format("YYYY-MM-DD HH:mm")}
          )
        </span>
      </footer>
    </div>
  );
}

function SummaryCard({ title, value }) {
  return (
    <div className="summary-card">
      <span className="summary-title">{title}</span>
      <span className="summary-value">{value ?? "—"}</span>
    </div>
  );
}

export default App;
