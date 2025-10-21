import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import dayjs from "dayjs";
import MonitoredUsersPanel from "./components/MonitoredUsersPanel";
import TweetFeed from "./components/TweetFeed";
import SettingsPanel from "./components/SettingsPanel";
import SearchBar from "./components/SearchBar";
import TweetList from "./components/TweetList";
import SentimentChart from "./components/SentimentChart";

const API_URL = "/api/tweets";
const LIVE_FETCH_URL = "/api/tweets/fetch";

const MAX_FEED_TWEETS = 4;

function normaliseHandle(value) {
  return value.trim().replace(/^@+/, "");
}

function App() {
  const [activeView, setActiveView] = useState("dashboard");

  const [monitoredUsers, setMonitoredUsers] = useState([]);
  const [newUserInput, setNewUserInput] = useState("");
  const [accountsNotice, setAccountsNotice] = useState(null);
  const [accountsError, setAccountsError] = useState(null);

  const [monitoredTweets, setMonitoredTweets] = useState([]);
  const [feedLoading, setFeedLoading] = useState(false);
  const [feedError, setFeedError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const [sortKey, setSortKey] = useState("date");
  const [sortDir, setSortDir] = useState("desc");
  const [refreshMinutes, setRefreshMinutes] = useState(60);
  const [marketSignalThreshold, setMarketSignalThreshold] = useState(40);

  const [searchHandle, setSearchHandle] = useState("");
  const [searchTweets, setSearchTweets] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState(null);
  const [searchNotice, setSearchNotice] = useState(null);
  const monitorFallbackTimeoutRef = useRef(null);
  const monitoredUsersRef = useRef([]);
  const initialFetchDoneRef = useRef(false);

  useEffect(() => {
    monitoredUsersRef.current = monitoredUsers;
  }, [monitoredUsers]);

  const dedupeHandles = useCallback((handles) => {
    const seen = new Set();
    const cleaned = [];
    handles.forEach((value) => {
      const handle = normaliseHandle(value);
      const key = handle.toLowerCase();
      if (handle && !seen.has(key)) {
        seen.add(key);
        cleaned.push(handle);
      }
    });
    return cleaned;
  }, []);

  const syncAccounts = useCallback(
    (nextUsers, previousUsers, options = {}) => {
      const { onPersisted } = options;
      const cleaned = dedupeHandles(nextUsers);
      setMonitoredUsers(cleaned);
      setAccountsNotice(null);
      setAccountsError(null);

      (async () => {
        try {
          const response = await fetch("/api/accounts", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ accounts: cleaned }),
          });

          if (!response.ok) {
            let detail = `Request failed (${response.status})`;
            try {
              const payload = await response.json();
              if (payload?.detail) detail = payload.detail;
            } catch (error) {
              // ignore JSON parse errors
            }
            throw new Error(detail);
          }

          const data = await response.json();
          const persisted = dedupeHandles(data.accounts ?? cleaned);
          setMonitoredUsers(persisted);
          setAccountsNotice(
            `Watch list saved (${persisted.length} handle${persisted.length === 1 ? "" : "s"}).`,
          );
          if (typeof onPersisted === "function") {
            const previousSet = new Set(
              (previousUsers || []).map((user) => normaliseHandle(user).toLowerCase()),
            );
            const added = persisted.filter(
              (handle) => !previousSet.has(handle.toLowerCase()),
            );
            onPersisted({ persisted, added });
          }
        } catch (error) {
          if (previousUsers) {
            setMonitoredUsers(previousUsers);
          }
          setAccountsError(error.message || "Failed to update watch list.");
          setAccountsNotice(null);
        }
      })();
    },
    [dedupeHandles],
  );

  useEffect(() => {
    let cancelled = false;

    const loadAccounts = async () => {
      try {
        const response = await fetch("/api/accounts");
        if (!response.ok) {
          throw new Error(`API error (${response.status})`);
        }
        const data = await response.json();
        const accounts = dedupeHandles(data.accounts ?? []);
        if (cancelled) return;
        setMonitoredUsers(accounts);
      } catch (error) {
        if (cancelled) return;
        setMonitoredUsers([]);
        setAccountsError(error.message || "Failed to load watch list.");
      }
    };

    loadAccounts();

    return () => {
      cancelled = true;
    };
  }, [dedupeHandles, syncAccounts]);

  const fetchMonitoredTweets = useCallback(
    async (handles, options = {}) => {
      const { replace = !handles } = options;
      const list = Array.isArray(handles) ? handles : null;
      const usersToFetch = list && list.length ? dedupeHandles(list) : monitoredUsersRef.current;

      if (!usersToFetch.length) {
        if (replace) {
          setMonitoredTweets([]);
          setFeedError(null);
        }
        return;
      }

      if (replace) {
        setFeedLoading(true);
      }
      setFeedError(null);

      const combined = [];
      const errors = [];

      try {
        await Promise.all(
          usersToFetch.map(async (user) => {
            const handle = normaliseHandle(user);
            if (!handle) return;
            try {
              const response = await fetch(
                `${API_URL}?username=${encodeURIComponent(handle)}`,
              );
              if (!response.ok) {
                throw new Error(`API error (${response.status})`);
              }
              const data = await response.json();
              const tweets = data.tweets ?? [];
              tweets.forEach((tweet) => {
                combined.push({ ...tweet, source_user: handle });
              });
            } catch (error) {
              errors.push(`@${handle}`);
            }
          }),
        );

        if (errors.length) {
          setFeedError(`Failed to refresh: ${errors.join(", ")}`);
        } else {
          setFeedError(null);
        }

        const lowerCaseUsers = new Set(
          usersToFetch
            .map((user) => normaliseHandle(user).toLowerCase())
            .filter(Boolean),
        );
        const dedupeTweets = (tweets) => {
          const seen = new Set();
          return tweets.filter((tweet) => {
            const key = tweet.tweet_id || `${tweet.source_user}-${tweet.created_at || tweet.text}`;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
          });
        };
        const dedupedCombined = dedupeTweets(combined);

        if (replace) {
          setMonitoredTweets(dedupedCombined);
        } else {
          setMonitoredTweets((prev) => {
            const remaining = prev.filter(
              (tweet) => !lowerCaseUsers.has((tweet.source_user || "").toLowerCase()),
            );
            const merged = [...remaining, ...dedupedCombined];
            return dedupeTweets(merged);
          });
        }

        setLastUpdated(dayjs().format("YYYY-MM-DD HH:mm"));
      } catch (error) {
        setFeedError(error.message || "Failed to refresh monitored tweets.");
      } finally {
        if (replace) {
          setFeedLoading(false);
        }
      }
    },
    [dedupeHandles],
  );

  const triggerMonitorRun = useCallback(async () => {
    try {
      await fetch("/api/monitor/run", { method: "POST" });
    } catch (error) {
      console.warn("Failed to trigger monitor run", error);
    }
  }, []);

  useEffect(() => {
    if (!initialFetchDoneRef.current && monitoredUsers.length) {
      initialFetchDoneRef.current = true;
      fetchMonitoredTweets();
    }
  }, [monitoredUsers, fetchMonitoredTweets]);

  useEffect(() => {
    if (!refreshMinutes || refreshMinutes <= 0) return undefined;
    const execute = () => {
      triggerMonitorRun();
      if (monitorFallbackTimeoutRef.current) {
        clearTimeout(monitorFallbackTimeoutRef.current);
      }
      monitorFallbackTimeoutRef.current = setTimeout(fetchMonitoredTweets, 15000);
    };

    execute();
    const interval = setInterval(execute, refreshMinutes * 60 * 1000);
    return () => {
      clearInterval(interval);
      if (monitorFallbackTimeoutRef.current) {
        clearTimeout(monitorFallbackTimeoutRef.current);
        monitorFallbackTimeoutRef.current = null;
      }
    };
  }, [triggerMonitorRun, fetchMonitoredTweets, refreshMinutes]);

  useEffect(() => {
    if (typeof window === "undefined" || !("WebSocket" in window)) {
      return undefined;
    }

    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${window.location.host}/ws/monitor`);
    socket.onmessage = () => {
      if (monitorFallbackTimeoutRef.current) {
        clearTimeout(monitorFallbackTimeoutRef.current);
        monitorFallbackTimeoutRef.current = null;
      }
      fetchMonitoredTweets();
    };
    socket.onerror = () => {
      socket.close();
    };

    return () => {
      socket.close();
    };
  }, [fetchMonitoredTweets]);

  const feedTweets = useMemo(() => {
    const threshold = Number(marketSignalThreshold) / 100;
    const filtered = monitoredTweets.filter((tweet) => {
      if (!Number.isFinite(threshold)) return true;
      const probability = Number(tweet.market_related_prob ?? 0);
      return probability >= threshold;
    });

    const toNumber = (value, fallback = 0) => {
      const num = Number(value);
      return Number.isFinite(num) ? num : fallback;
    };

    const sorted = filtered.sort((a, b) => {
      const direction = sortDir === "asc" ? 1 : -1;
      switch (sortKey) {
        case "market": {
          const diff = toNumber(a.market_related_prob) - toNumber(b.market_related_prob);
          return diff * direction;
        }
        case "confidence": {
          const diff = toNumber(a.sentiment_prob) - toNumber(b.sentiment_prob);
          return diff * direction;
        }
        case "engagement": {
          const diff = toNumber(a.engagement) - toNumber(b.engagement);
          return diff * direction;
        }
        case "date":
        default: {
          const diff = new Date(a.created_at ?? 0) - new Date(b.created_at ?? 0);
          return diff * direction;
        }
      }
    });

    return sorted.slice(0, MAX_FEED_TWEETS);
  }, [monitoredTweets, marketSignalThreshold, sortKey, sortDir]);

  const handleAddMonitoredUser = () => {
    const handle = normaliseHandle(newUserInput);
    setNewUserInput("");
    if (!handle) {
      return;
    }

    const exists = monitoredUsers.some((user) => user.toLowerCase() === handle.toLowerCase());
    if (exists) {
      return;
    }

    const next = [...monitoredUsers, handle];
    syncAccounts(next, monitoredUsers, {
      onPersisted: ({ added }) => {
        if (Array.isArray(added) && added.length) {
          fetchMonitoredTweets(added, { replace: false });
        }
      },
    });
  };

  const handleRemoveUser = (user) => {
    const next = monitoredUsers.filter((item) => item.toLowerCase() !== user.toLowerCase());
    syncAccounts(next, monitoredUsers, {
      onPersisted: ({ persisted }) => {
        const allowed = new Set(persisted.map((handle) => handle.toLowerCase()));
        setMonitoredTweets((prev) =>
          prev.filter((tweet) => allowed.has((tweet.source_user || "").toLowerCase())),
        );
      },
    });
  };

  const handleFetchOnDemand = async () => {
    const handle = normaliseHandle(searchHandle);
    if (!handle) {
      setSearchError("Enter a handle to fetch tweets.");
      setSearchNotice(null);
      return;
    }

    setSearchLoading(true);
    setSearchError(null);
    setSearchNotice(null);

    try {
      const response = await fetch(LIVE_FETCH_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: handle, limit: MAX_FEED_TWEETS }),
      });

      if (!response.ok) {
        let detail = `Request failed (${response.status})`;
        try {
          const payload = await response.json();
          if (payload?.detail) detail = payload.detail;
        } catch (error) {
          // ignore parse errors
        }
        throw new Error(detail);
      }

      const data = await response.json();
      const tweets = data?.tweets ?? [];
      setSearchTweets(tweets);
      setSearchNotice(
        tweets.length
          ? `Fetched ${tweets.length} tweet${tweets.length === 1 ? "" : "s"} for @${handle}.`
          : `No tweets found for @${handle}.`,
      );
    } catch (error) {
      setSearchTweets([]);
      setSearchError(error.message || "Unable to fetch tweets.");
    } finally {
      setSearchLoading(false);
    }
  };

  return (

    <div className="page-shell">
      <script src="http://localhost:8097"></script>
      <header className="hero">
        <div className="brand">
          <span className="brand__icon" aria-hidden="true" />
          <div>
            <h1>Market Tweet Monitor</h1>
            <p className="tagline">
              Track market sentiment and engagement from top financial voices.
            </p>
          </div>
        </div>

        <nav className="tabs">
          <button
            type="button"
            className={activeView === "dashboard" ? "active" : ""}
            onClick={() => setActiveView("dashboard")}
          >
            Monitoring Dashboard
          </button>
          <button
            type="button"
            className={activeView === "ondemand" ? "active" : ""}
            onClick={() => setActiveView("ondemand")}
          >
            On-Demand Fetch
          </button>
        </nav>
      </header>

      {activeView === "dashboard" ? (
        <>
          <div className="dashboard-grid">
            <div className="panel-stack">
              <MonitoredUsersPanel
                users={monitoredUsers}
                newUser={newUserInput}
                onChangeNewUser={setNewUserInput}
                onAddUser={handleAddMonitoredUser}
                onRemoveUser={handleRemoveUser}
              />
              {(accountsNotice || accountsError) && (
                <p className={`helper-text ${accountsError ? "error" : "success"}`}>
                  {accountsError || accountsNotice}
                </p>
              )}
            </div>
            <SettingsPanel
              sortKey={sortKey}
              sortDir={sortDir}
              refreshMinutes={refreshMinutes}
              marketSignalThreshold={marketSignalThreshold}
              onChangeSortKey={setSortKey}
              onChangeSortDir={setSortDir}
              onChangeRefresh={setRefreshMinutes}
              onChangeMarketSignalThreshold={setMarketSignalThreshold}
            />
          </div>

          <TweetFeed
            tweets={feedTweets}
            loading={feedLoading}
            error={feedError}
            lastUpdated={lastUpdated}
          />
        </>
      ) : (
        <section className="on-demand">
          <SearchBar
            value={searchHandle}
            onChange={setSearchHandle}
            onSubmit={handleFetchOnDemand}
            loading={searchLoading}
          />
          {(searchNotice || searchError) && (
            <p className={`helper-text ${searchError ? "error" : "success"}`}>
              {searchError || searchNotice}
            </p>
          )}
          {!searchTweets.length && !searchLoading && !searchError ? (
            <section className="panel panel-placeholder">
              <p>Enter a handle and click &quot;Fetch Tweets&quot; to analyze.</p>
            </section>
          ) : (
            <>
              <SentimentChart tweets={searchTweets} />
              <TweetList tweets={searchTweets} loading={searchLoading} error={searchError} />
            </>
          )}
        </section>
      )}

      <footer className="footer">
        <span>
          Data source refreshed at {lastUpdated ?? dayjs().format("YYYY-MM-DD HH:mm")}
        </span>
      </footer>
    </div>
  );
}

export default App;
