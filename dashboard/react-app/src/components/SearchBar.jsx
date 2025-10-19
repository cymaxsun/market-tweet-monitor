import React from "react";

function normaliseHandle(value) {
  return value.trim().replace(/^@+/, "");
}

export default function SearchBar({ value, onChange, onSubmit, loading }) {
  const disabled = loading || !normaliseHandle(value);

  return (
    <section className="panel search-panel">
      <header className="panel-header">
        <h2>Search Twitter Handle</h2>
        <p>Look up fresh sentiment on demand for any account.</p>
      </header>

      <div className="search-bar">
        <label htmlFor="search-handle" className="search-bar__field">
          <span>Enter handle (e.g., elonmusk)</span>
          <input
            id="search-handle"
            type="text"
            placeholder="elonmusk"
            value={value}
            onChange={(event) => onChange(normaliseHandle(event.target.value))}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !disabled) {
                event.preventDefault();
                onSubmit();
              }
            }}
          />
        </label>
        <button
          type="button"
          className="primary-btn"
          onClick={onSubmit}
          disabled={disabled}
        >
          {loading ? "Fetching…" : "Fetch Tweets"}
        </button>
      </div>
    </section>
  );
}
