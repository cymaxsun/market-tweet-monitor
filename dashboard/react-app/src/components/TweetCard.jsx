import { useMemo, useState } from "react";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";

dayjs.extend(relativeTime);

const formatNumber = (value) => {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number" && !Number.isInteger(value)) {
    return value.toFixed(2);
  }
  return value.toLocaleString?.() ?? value;
};

const formatProbability = (value) => {
  if (value === null || value === undefined) return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return `${Math.round(number * 100)}%`;
};

const resolveSentimentState = (value) => {
  if (value === null || value === undefined) {
    return { label: "Neutral", tone: "neutral" };
  }
  if (value == "positive") return { label: "Positive", tone: "bullish" };
  if (value == "negative") return { label: "Negative", tone: "bearish" };
  return { label: "Neutral", tone: "neutral" };
};

const sentimentToneToClass = {
  positive: "tweet-card__badge--bullish",
  negative: "tweet-card__badge--bearish",
  neutral: "tweet-card__badge--neutral",
};

const ACTIONS = [
  { label: "Likes", key: "likes", icon: "♥" },
  { label: "Retweets", key: "retweets", icon: "⟳" },
  { label: "Replies", key: "replies", icon: "💬" },
];

const MAX_PREVIEW_CHARACTERS = 260;
const MARKET_SIGNAL_HIGH_THRESHOLD = 0.75;
const MARKET_SIGNAL_LOW_THRESHOLD = 0.35;
const HIGH_ENGAGEMENT_THRESHOLD = 1000;

export default function TweetCard({ tweet }) {
  const createdAt = tweet.created_at ? dayjs(tweet.created_at) : null;
  const relative = createdAt ? createdAt.fromNow(true) : null;
  const sentimentState = resolveSentimentState(tweet.sentiment);
  const retweetedUser = tweet.retweeted_user;
  const isRetweet = Boolean(tweet.is_retweet);
  const rawContent = (tweet.cleaned || tweet.text || "").trim();
  const [expanded, setExpanded] = useState(false);

  const { displayLines, showToggle } = useMemo(() => {
    if (!rawContent) {
      return { displayLines: ["No text available."], showToggle: false };
    }

    const isLong = rawContent.length > MAX_PREVIEW_CHARACTERS;
    const text = expanded || !isLong
      ? rawContent
      : `${rawContent.slice(0, MAX_PREVIEW_CHARACTERS).trimEnd()}…`;

    return { displayLines: text.split(/\r?\n/), showToggle: isLong };
  }, [expanded, rawContent]);

  const marketProbability = Number(tweet.market_related_prob);
  const marketToneClass = Number.isFinite(marketProbability)
    ? marketProbability >= MARKET_SIGNAL_HIGH_THRESHOLD
      ? "tweet-card__stat--market-high"
      : marketProbability <= MARKET_SIGNAL_LOW_THRESHOLD
        ? "tweet-card__stat--market-low"
        : ""
    : "";

  const engagementScore = Number(tweet.engagement);
  const hasHighEngagement =
    Number.isFinite(engagementScore) && engagementScore >= HIGH_ENGAGEMENT_THRESHOLD;
  const engagementToneClass = hasHighEngagement ? "tweet-card__stat--engagement-high" : "";
  const engagementFlair = hasHighEngagement
    ? { icon: "🔥", text: "High engagement", ariaLabel: "High engagement" }
    : null;

  return (
    <article className="tweet-card tweet-card--modern">
      <header className="tweet-card__top">
        <div className="tweet-card__identity">
          <span className="tweet-card__user">@{tweet.username ?? "unknown"}</span>
          <span className="tweet-card__time">{relative ? `${relative} ago` : "Just now"}</span>
          {isRetweet && (
            <span className="tweet-card__retweet-banner">
              <span className="tweet-card__retweet-icon" aria-hidden="true">
                ↻
              </span>
              <span className="tweet-card__retweet-text">
                {retweetedUser ? `Retweeted from @${retweetedUser}` : "Retweet"}
              </span>
            </span>
          )}
        </div>
        <span className={`tweet-card__badge ${sentimentToneToClass[tweet.sentiment]}`}>
          {tweet.sentiment}
        </span>
      </header>

      <div className="tweet-card__body tweet-card__body--modern">
        {displayLines.map((line, idx) => (
          <span key={`${tweet.tweet_id ?? idx}-${idx}`} className="tweet-card__content-line">
            {line}
            {idx < displayLines.length - 1 && <br />}
          </span>
        ))}
        {showToggle && (
          <button
            type="button"
            className="tweet-card__content-toggle"
            onClick={() => setExpanded((prev) => !prev)}
          >
            {expanded ? "Show less" : "Read more"}
          </button>
        )}
      </div>

      <div className="tweet-card__stats">
        <Stat
          label="Sentiment Confidence"
          value={formatProbability(tweet.sentiment_prob)}
        />
        <Stat
          label="Market Signal"
          value={formatProbability(tweet.market_related_prob)}
          toneClass={marketToneClass}
        />
        <Stat
          label="Engagement"
          value={formatNumber(tweet.engagement)}
          toneClass={engagementToneClass}
          flair={engagementFlair}
        />
      </div>

      <footer className="tweet-card__footer tweet-card__footer--modern">
        {ACTIONS.map((action) => (
          <div key={action.key} className="tweet-card__action">
            <span className="tweet-card__action-icon" aria-hidden="true">
              {action.icon}
            </span>
            <span className="tweet-card__action-label">{action.label}</span>
            <span className="tweet-card__action-value">
              {formatNumber(tweet[action.key])}
            </span>
          </div>
        ))}
      </footer>
    </article>
  );
}

function Stat({ label, value, toneClass, flair }) {
  return (
    <div className={`tweet-card__stat ${toneClass ?? ""}`}>
      <div className="tweet-card__stat-top">
        <span className="tweet-card__stat-label">{label}</span>
        {flair ? (
          <span className="tweet-card__flair" aria-label={flair.ariaLabel ?? flair.text}>
            <span className="tweet-card__flair-icon" aria-hidden="true">
              {flair.icon}
            </span>
            <span className="tweet-card__flair-text">{flair.text}</span>
          </span>
        ) : null}
      </div>
      <span className="tweet-card__stat-value">{value}</span>
    </div>
  );
}
