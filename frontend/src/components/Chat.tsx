import { AgUiEvent, categoryOf, ChatTurn, EventType, SearchView } from "../lib/types";
import { Markdown } from "../lib/markdown";
import { useState } from "react";

export function Turn({ turn }: { turn: ChatTurn }) {
  const isUser = turn.role === "user";
  return (
    <div className={`turn ${turn.role}`}>
      <div className="avatar">{isUser ? "You" : "AI"}</div>
      <div className="turn-body">
        <div className="who">{isUser ? "You" : "Assistant"}</div>
        {isUser ? (
          <div className="user-text">{turn.content}</div>
        ) : (
          <>
            {(turn.thinking || turn.thinkingActive) && (
              <Thinking text={turn.thinking} active={turn.thinkingActive} />
            )}
            {turn.searches.map((s) => (
              <SearchCard key={s.id} search={s} />
            ))}
            {turn.content && <Markdown text={turn.content} />}
            {turn.streaming && !turn.content && !turn.thinkingActive && turn.searches.length === 0 && (
              <span className="cursor" />
            )}
            {turn.streaming && turn.content && <span className="cursor" />}
          </>
        )}
      </div>
    </div>
  );
}

function Thinking({ text, active }: { text: string; active: boolean }) {
  const [open, setOpen] = useState(true);
  return (
    <div className={`thinking ${active ? "active" : ""}`}>
      <button className="thinking-head" onClick={() => setOpen((o) => !o)}>
        <span className="spark">{active ? "✦" : "✓"}</span>
        {active ? "Thinking…" : "Thought process"}
        <span className="chev">{open ? "▾" : "▸"}</span>
      </button>
      {open && text && <div className="thinking-body">{text}</div>}
    </div>
  );
}

function SearchCard({ search }: { search: SearchView }) {
  return (
    <div className="search-card">
      <div className="search-head">
        <span className="search-icon">🔍</span>
        <span className="search-q">
          {search.status === "searching" ? "Searching the web" : "Searched the web"}
          {search.query ? `: "${search.query}"` : "…"}
        </span>
        <span className={`search-status ${search.status}`}>{search.status}</span>
      </div>
      {search.results.length > 0 && (
        <div className="search-results">
          {search.results.slice(0, 5).map((r, i) => (
            <a
              key={i}
              className="search-result"
              href={r.url || undefined}
              target="_blank"
              rel="noopener noreferrer"
            >
              <span className="sr-title">{r.title || r.url}</span>
              <span className="sr-snippet">{r.snippet}</span>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

const CATS = ["lifecycle", "thinking", "text", "tool", "state", "custom", "raw"];

function detail(ev: AgUiEvent): string {
  switch (ev.type) {
    case EventType.RUN_STARTED:
    case EventType.RUN_FINISHED:
      return `${ev.threadId ?? ""} / ${ev.runId ?? ""}`;
    case EventType.RUN_ERROR:
      return ev.message ?? "";
    case EventType.STEP_STARTED:
    case EventType.STEP_FINISHED:
      return ev.stepName ?? "";
    case EventType.TEXT_MESSAGE_CONTENT:
      return JSON.stringify(ev.delta);
    case EventType.TOOL_CALL_START:
      return ev.toolCallName ?? "";
    case EventType.TOOL_CALL_ARGS:
      return String(ev.delta ?? "");
    case EventType.TOOL_CALL_RESULT:
      return (ev.content ?? "").slice(0, 120);
    case EventType.CUSTOM:
      return `${ev.name}: ${JSON.stringify(ev.value)}`;
    case EventType.RAW:
      return `${ev.source}: ${JSON.stringify(ev.event)}`;
    case EventType.STATE_SNAPSHOT:
      return JSON.stringify(ev.snapshot);
    case EventType.STATE_DELTA:
      return JSON.stringify(ev.delta);
    default:
      return ev.messageId ?? "";
  }
}

export function EventStream({ events, onClear }: { events: AgUiEvent[]; onClear: () => void }) {
  return (
    <>
      <div className="stream-head">
        Event stream — real ag_ui.core events
        <button className="mini" onClick={onClear}>clear</button>
        <span className="count">{events.length}</span>
      </div>
      <div className="legend">
        {CATS.map((c) => (
          <span key={c}>
            <i style={{ background: `var(--c-${c})` }} />
            {c}
          </span>
        ))}
      </div>
      <div className="stream-list">
        {events.length === 0 && <div className="stream-empty">No events yet. Send a message.</div>}
        {events.map((ev, i) => {
          const cat = categoryOf(ev.type, ev.name);
          return (
            <div className={`erow cat-${cat}`} key={i}>
              <span className="enm">{ev.type}</span>
              <span className="edt">{detail(ev)}</span>
            </div>
          );
        })}
      </div>
    </>
  );
}
