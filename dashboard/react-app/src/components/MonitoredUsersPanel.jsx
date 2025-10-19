import React from "react";

function normaliseHandle(value) {
  return value.trim().replace(/^@+/, "");
}

export default function MonitoredUsersPanel({
  users,
  newUser,
  onChangeNewUser,
  onAddUser,
  onRemoveUser,
}) {
  const filterTerm = newUser ? newUser.toLowerCase() : "";
  const filteredUsers =
    filterTerm.length > 0
      ? users.filter((user) => user.toLowerCase().includes(filterTerm))
      : users;

  let listContent = null;
  if (!users.length) {
    listContent = <p className="muted">No accounts tracked yet. Add a handle above.</p>;
  } else if (filterTerm && !filteredUsers.length) {
    listContent = <p className="muted">No accounts match "{newUser}".</p>;
  } else {
    listContent = (
      <ul className="monitored-users__list">
        {filteredUsers.map((user) => (
          <li key={user}>
            <span>@{user}</span>
            <button
              type="button"
              onClick={() => onRemoveUser(user)}
              aria-label={`Remove @${user}`}
            >
              ✕
            </button>
          </li>
        ))}
      </ul>
    );
  }

  return (
    <section className="panel monitored-users">
      <header className="panel-header">
        <h2>Monitored Accounts</h2>
        <p>Keep tabs on the voices that move markets first.</p>
      </header>

      <div className="monitored-users__input">
        <input
          type="text"
          placeholder="Filter or add handle..."
          value={newUser}
          onChange={(event) => onChangeNewUser(normaliseHandle(event.target.value))}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              onAddUser();
            }
          }}
        />
        <button
          type="button"
          className="primary-btn primary-btn--icon"
          onClick={onAddUser}
          disabled={!normaliseHandle(newUser)}
          aria-label="Add handle"
        >
          <span>+</span>
        </button>
      </div>

      {listContent}
    </section>
  );
}
