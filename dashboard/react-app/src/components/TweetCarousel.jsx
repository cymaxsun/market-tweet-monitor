import { useEffect, useState } from "react";
import dayjs from "dayjs";

const FALLBACK_TEXT = "No tweets match the current filters.";

const formatNumber = (value) => {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number" && !Number.isInteger(value)) {
    return value.toFixed(2);
  }
  return value.toLocaleString?.() ?? value;
};

export default function TweetCarousel({ tweets, loading }) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    setIndex(0);
  }, [tweets]);

  if (loading) {
    return (
      <section className="card-loading">
        <div className="spinner" />
        <p>Loading tweets...</p>
      </section>
    );
  }

  if (!tweets.length) {
    return (
      <section className="card-empty">
        <p>{FALLBACK_TEXT}</p>
      </section>
    );
  }

  const tweet = tweets[index];
  const createdAt = tweet.created_at ? dayjs(tweet.created_at).format("YYYY-MM-DD HH:mm") : "Unknown";
  const lines = tweet.cleaned ? tweet.cleaned.split(/\r?\n/) : [];

  const goPrev = () => setIndex((prev) => Math.max(prev - 1, 0));
  const goNext = () => setIndex((prev) => Math.min(prev + 1, tweets.length - 1));

  return (
    <section className="carousel-shell">
      <div className="carousel-nav">
        <button
          type="button"
          onClick={goPrev}
          disabled={index === 0}
          aria-label="Previous tweet"
        >
          ◀
        </button>
      </div>

      <article className="tweet-card">
        <div className="tweet-card-content">
          <header className="tweet-header">
            <span>@{tweet.username}</span>
            <span>{createdAt}</span>
          </header>
          <div className="tweet-body">
            {lines.length
              ? lines.map((line, idx) => (
                  <span key={`${tweet.tweet_id ?? idx}-${idx}`}>
                    {line}
                    {idx < lines.length - 1 && <br />}
                  </span>
                ))
              : "No text available."}
          </div>
          <div className="tweet-metrics">
            <Metric label="Sentiment" value={formatNumber(tweet.sentiment)} />
            <Metric label="Engagement" value={formatNumber(tweet.engagement)} />
            <Metric label="Likes" value={formatNumber(tweet.likes)} />
            <Metric label="Retweets" value={formatNumber(tweet.retweets)} />
            <Metric label="Replies" value={formatNumber(tweet.replies)} />
          </div>
        </div>
      </article>

      <div className="carousel-nav">
        <button
          type="button"
          onClick={goNext}
          disabled={index === tweets.length - 1}
          aria-label="Next tweet"
        >
          ▶
        </button>
      </div>

      <p className="carousel-status">
        Tweet {index + 1} of {tweets.length}
      </p>
    </section>
  );
}

function Metric({ label, value }) {
  return (
    <span>
      <strong>{label}</strong>
      {value}
    </span>
  );
}
