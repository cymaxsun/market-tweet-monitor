import React from "react";

const SORT_OPTIONS = [
  { value: "date", label: "Date" },
  { value: "market", label: "Market Signal" },
  { value: "confidence", label: "Sentiment Confidence" },
  { value: "engagement", label: "Engagement" },
];

export default function SettingsPanel({
  sortKey,
  sortDir,
  refreshMinutes,
  marketSignalThreshold,
  onChangeSortKey,
  onChangeSortDir,
  onChangeRefresh,
  onChangeMarketSignalThreshold,
}) {
  return (
    <section className="panel settings-panel">
      <header className="panel-header">
        <h2>Settings</h2>
        <p>Tune the feed so only the sharpest signals cut through.</p>
      </header>

      <div className="settings-grid">
        <label className="settings-field">
          <span>Sort By</span>
          <select value={sortKey} onChange={(event) => onChangeSortKey(event.target.value)}>
            {SORT_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        <label className="settings-field">
          <span>Sort Direction</span>
          <select value={sortDir} onChange={(event) => onChangeSortDir(event.target.value)}>
            <option value="desc">Descending</option>
            <option value="asc">Ascending</option>
          </select>
        </label>

        <label className="settings-field">
          <span>Refresh Interval</span>
          <select value={refreshMinutes} onChange={(event) => onChangeRefresh(Number(event.target.value))}>
            <option value={30}>30 minutes</option>
            <option value={60}>1 hour</option>
            <option value={120}>2 hours</option>
            <option value={180}>3 hours</option>
            <option value={240}>4 hours</option>
          </select>
          <small className="muted">Monitor + dashboard refresh cadence.</small>
        </label>

        <label className="settings-field">
          <span>Market Signal Threshold</span>
          <input
            type="range"
            min="0"
            max="100"
            step="5"
            value={marketSignalThreshold}
            onChange={(event) => onChangeMarketSignalThreshold(Number(event.target.value))}
          />
          <strong>{marketSignalThreshold}%</strong>
          <small className="muted">
            Only show tweets with a market-related probability at or above this percentage.
          </small>
        </label>
      </div>
    </section>
  );
}
