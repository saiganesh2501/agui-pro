import { useEffect, useRef, useState } from "react";
import { useAgent } from "./hooks/useAgent";
import { Turn, EventStream } from "./components/Chat";
import { MODELS } from "./lib/types";

export default function App() {
  const [model, setModel] = useState<string>(MODELS[0]);
  const {
    status, turns, events, generating, error,
    sendMessage, stop, inject, regenerate, newChat, clearEvents,
  } = useAgent(model);

  const [text, setText] = useState("");
  const [showStream, setShowStream] = useState(true);
  const threadRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight, behavior: "smooth" });
  }, [turns]);

  const onSubmit = () => {
    const t = text.trim();
    if (!t || status !== "open") return;
    if (generating) {
      // while generating, treat the input as an injected instruction
      inject(t);
    } else {
      sendMessage(t);
    }
    setText("");
  };

  const canRegen = turns.some((t) => t.role === "assistant") && !generating;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="bname">AG-UI</span>
          <span className="btag">streaming agent · official protocol · websocket</span>
        </div>
        <div className="spacer" />
        <select className="select" value={model} onChange={(e) => setModel(e.target.value)} disabled={generating}>
          {MODELS.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
        <button className="btn" onClick={newChat} disabled={generating}>New chat</button>
        <button className="btn" onClick={() => setShowStream((s) => !s)}>
          {showStream ? "Hide events" : "Show events"}
        </button>
        <div className="conn">
          <span className={`dot ${status}`} />
          {status === "open" ? "connected" : status === "connecting" ? "connecting…" : "reconnecting…"}
        </div>
      </header>

      {error && <div className="errbar">⚠️ {error}</div>}

      <div className="main">
        <div className="chat">
          <div className="thread" ref={threadRef}>
            <div className="thread-inner">
              {turns.length === 0 && (
                <div className="welcome">
                  <h1>What can I help with?</h1>
                  <p>
                    Try “Explain async/await with a code example”, or ask something current like
                    “What are the latest AG-UI protocol updates?” to watch the web-search tool run.
                  </p>
                </div>
              )}
              {turns.map((t) => (
                <Turn key={t.id} turn={t} />
              ))}
            </div>
          </div>

          <div className="composer-wrap">
            <div className="composer">
              {canRegen && (
                <button className="regen" onClick={regenerate}>↻ Regenerate</button>
              )}
              {generating && (
                <div className="gen-hint">
                  Generating… you can type to <b>inject instructions</b> into this run, or press ■ to stop.
                </div>
              )}
              <div className="composer-box">
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  rows={1}
                  placeholder={
                    status !== "open"
                      ? "Connecting to server…"
                      : generating
                      ? "Add instructions to the running task…"
                      : "Message the agent…"
                  }
                  onInput={(e) => {
                    const el = e.currentTarget;
                    el.style.height = "auto";
                    el.style.height = Math.min(el.scrollHeight, 200) + "px";
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      onSubmit();
                    }
                  }}
                />
                {generating ? (
                  <button className="iconbtn stop" onClick={stop} title="Stop generating">■</button>
                ) : (
                  <button
                    className="iconbtn send"
                    onClick={onSubmit}
                    disabled={!text.trim() || status !== "open"}
                    title="Send"
                  >
                    ↑
                  </button>
                )}
              </div>
              <div className="composer-foot">
                <span><b>Enter</b> send</span>
                <span><b>Shift+Enter</b> newline</span>
                <span><b>■</b> stop</span>
                <span>while generating, <b>Enter</b> injects instructions</span>
              </div>
            </div>
          </div>
        </div>

        {showStream && (
          <aside className="streampanel">
            <EventStream events={events} onClear={clearEvents} />
          </aside>
        )}
      </div>
    </div>
  );
}
