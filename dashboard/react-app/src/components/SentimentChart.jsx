export default function SentimentChart({ tweets }) {
  if (!tweets.length) {
    return null;
  }

  const sentiments = tweets
    .map((tweet) => Number(tweet.sentiment ?? 0))
    .filter((value) => Number.isFinite(value));

  if (!sentiments.length) {
    return null;
  }

  const avgSentiment =
    sentiments.reduce((total, value) => total + value, 0) / sentiments.length;
  const clamped = Math.max(-1, Math.min(1, avgSentiment));
  const percentage = ((clamped + 1) / 2) * 100;

  const sentimentLabel =
    clamped > 0.2 ? "Bullish" : clamped < -0.2 ? "Bearish" : "Neutral";

  return (
    <section className="panel sentiment-panel">
      <header className="panel-header">
        <h2>Sentiment Snapshot</h2>
        <span className="muted">
          Average sentiment across {sentiments.length} tweet
          {sentiments.length === 1 ? "" : "s"}
        </span>
      </header>

      <div className="sentiment-bar">
        <div className="sentiment-bar__axis sentiment-bar__axis--left">Bearish</div>
        <div className="sentiment-bar__track">
          <div
            className="sentiment-bar__fill"
            style={{ width: `${percentage}%` }}
          />
          <span className="sentiment-bar__marker" style={{ left: `${percentage}%` }} />
        </div>
        <div className="sentiment-bar__axis sentiment-bar__axis--right">Bullish</div>
      </div>

      <div className="sentiment-meta">
        <span className="sentiment-meta__label">{sentimentLabel}</span>
        <span className="sentiment-meta__score">
          Avg sentiment: {avgSentiment.toFixed(2)}
        </span>
      </div>
    </section>
  );
}
