import dayjs from "dayjs";
import TweetCard from "./TweetCard";

export default function TweetFeed({ tweets, loading, error, lastUpdated }) {
  if (loading) {
    return (
      <section className="panel tweet-feed">
        <div className="panel-loading">
          <div className="spinner" />
          <p>Loading monitored tweets…</p>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="panel tweet-feed">
        <div className="panel-error">
          <strong>Couldn&apos;t load tweets.</strong>
          <span>{error}</span>
        </div>
      </section>
    );
  }

  if (!tweets.length) {
    return (
      <section className="panel tweet-feed">
        <div className="panel-empty">
          <p>No tweets match the current filters.</p>
        </div>
      </section>
    );
  }

  const formattedUpdate = lastUpdated
    ? dayjs(lastUpdated).format("h:mm:ss A")
    : dayjs().format("h:mm:ss A");

  return (
    <section className="panel tweet-feed">
      <header className="panel-header panel-header--split">
        <div>
          <h2>Top Market Signals</h2>
          <span className="muted">
            {tweets.length} curated signal{tweets.length === 1 ? "" : "s"} from tracked accounts
          </span>
        </div>
        <span className="panel-meta">Last updated: {formattedUpdate}</span>
      </header>

      <div className="tweet-feed__list">
        {tweets.map((tweet) => (
          <TweetCard key={tweet.tweet_id ?? tweet.created_at} tweet={tweet} />
        ))}
      </div>
    </section>
  );
}
