import TweetCard from "./TweetCard";

export default function TweetList({ tweets, loading, error }) {
  if (loading) {
    return (
      <section className="panel tweet-list">
        <div className="panel-loading">
          <div className="spinner" />
          <p>Fetching tweets…</p>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="panel tweet-list">
        <div className="panel-error">
          <strong>Couldn&apos;t fetch tweets.</strong>
          <span>{error}</span>
        </div>
      </section>
    );
  }

  if (!tweets.length) {
    return null;
  }

  return (
    <section className="panel tweet-list">
      <header className="panel-header panel-header--split">
        <div>
          <h2>Recent Tweets</h2>
          <span className="muted">Latest activity from your on-demand lookup</span>
        </div>
        <span className="panel-meta">
          {tweets.length} result{tweets.length === 1 ? "" : "s"}
        </span>
      </header>

      <div className="tweet-feed__list">
        {tweets.map((tweet) => (
          <TweetCard key={tweet.tweet_id ?? tweet.created_at} tweet={tweet} />
        ))}
      </div>
    </section>
  );
}
