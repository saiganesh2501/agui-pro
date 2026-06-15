# AG-UI Streaming Agent

A ChatGPT-style streaming chat agent built on the **official AG-UI protocol**
(`ag-ui-protocol` Python SDK), with **token streaming, visible thinking, web
search, mid-generation stop, and human-in-the-loop instruction injection**.

- **Frontend:** React + TypeScript
- **Backend:** FastAPI (Python)
- **Transport:** WebSocket (full-duplex)
- **LLM:** OpenAI (key via env var)
- **Protocol:** AG-UI — real `ag_ui.core` event classes

> **On "real protocol":** this uses the genuine `ag_ui.core` event types from the
> official `ag-ui-protocol` package — not hand-rolled copies. The one deliberate
> deviation: the official frontend client (`@ag-ui/client`) only speaks SSE, but
> this app needs full-duplex WebSockets for live stop + mid-task injection. So
> the **events are official**, the **transport is WebSocket**, and the frontend
> WS client is custom. This was a conscious trade to meet the interactivity
> requirements; an SSE-only variant using `@ag-ui/client` is possible but can't
> do live injection over one channel.

---

## 1. Architecture overview

```
┌────────────────────────────┐     WebSocket  ws://localhost:8000/ws     ┌──────────────────────────────┐
│  React + TypeScript          │ ───────────────────────────────────────▶ │  FastAPI                       │
│                              │   client → server:                        │                               │
│  useAgent() hook             │     { run } { stop } { inject }           │  /ws endpoint (main.py)       │
│   • WS + auto-reconnect      │                                           │   • Steering (cancel/inject)  │
│   • event → chat state       │ ◀─────────────────────────────────────── │   • StreamingAgent (agent.py) │
│   • stop / inject / regen    │   server → client:                        │       │                        │
│  Components: Turn, Thinking, │     official ag_ui.core events (camelCase)│       ▼                        │
│   SearchCard, EventStream    │                                           │   OpenAI (stream=True)        │
└────────────────────────────┘                                           │   web_search tool (tools.py)  │
                                                                          └──────────────────────────────┘
```

### Event flow for one message

1. User types → client sends `{ type: "run", input: RunAgentInput }`.
2. Server starts `StreamingAgent.run()` as a background task; emits `RUN_STARTED`,
   then a `STATE_SNAPSHOT`.
3. The agent calls OpenAI with `stream=True`. As tokens arrive:
   - Tool-planning tokens → `STEP_STARTED("thinking")` + `CUSTOM thinking.delta`
     (shown as the dashed "Thinking…" block).
   - If the model calls `web_search` → `TOOL_CALL_START / ARGS / END`, the tool
     runs, then `TOOL_CALL_RESULT` (shown as the search card with results).
   - Answer tokens → `TEXT_MESSAGE_START / CONTENT / END` (the streamed reply).
4. Loop repeats if the model requests more tools (up to `MAX_TOOL_ROUNDS`).
5. Server emits `RUN_FINISHED`. Client re-enables input.

### Stop-and-resume flow

- **Stop:** client sends `{ type: "stop" }`. Server flips `Steering.cancelled`
  **and closes the live OpenAI stream** (`await stream.close()`), so the in-flight
  request is dropped server-side — not just hidden. The agent unwinds and emits
  `RUN_ERROR(code="STOPPED")`. Verified in tests: only a few tokens stream before
  stop, and the OpenAI stream object is confirmed closed.
- **Inject after/while running:** type new text while generating → client sends
  `{ type: "inject", text }`. The server appends it to `Steering.injected_notes`;
  the agent folds it into the conversation at the **next round** as a user message
  and emits `CUSTOM inject.ack`. If idle, the same input is sent as a normal new
  message.

---

## 2. AG-UI event → UI state mapping

| Event | UI effect |
|-------|-----------|
| `RUN_STARTED` | `generating = true`, input switches to stop mode |
| `RUN_FINISHED` | `generating = false`, finalize turn, re-enable input |
| `RUN_ERROR` | stop spinner; show error (unless `code="STOPPED"`) |
| `STEP_STARTED("thinking")` / `STEP_FINISHED` | toggle the dashed "Thinking…" block |
| `CUSTOM thinking.delta` | append text into the thinking block |
| `TEXT_MESSAGE_START` | begin the assistant answer bubble |
| `TEXT_MESSAGE_CONTENT` | append a token to the answer (streaming) |
| `TEXT_MESSAGE_END` | mark the answer complete |
| `TOOL_CALL_START` (web_search) | add a search card in "searching" state |
| `TOOL_CALL_ARGS` | parse the streamed query into the card title |
| `TOOL_CALL_RESULT` | fill the card with results, mark "done" |
| `STATE_SNAPSHOT` / `STATE_DELTA` | run status / tool-round counter |
| `CUSTOM inject.ack` | confirms an injected instruction was queued |
| `RAW` | tool telemetry (shown in the event inspector) |

The right-hand **Event stream** panel shows every raw event live, color-coded.

---

## 3. Project structure

```
agui-pro/
├── backend/
│   ├── events.py          # re-exports official ag_ui.core event classes
│   ├── tools.py           # web_search tool (DuckDuckGo) + OpenAI schema
│   ├── agent.py           # StreamingAgent: OpenAI stream, thinking, tools, cancel
│   ├── main.py            # FastAPI WebSocket endpoint, stop/inject routing
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.ts
    ├── tsconfig.json
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── index.css
        ├── lib/
        │   ├── types.ts       # AG-UI event types (TS)
        │   └── markdown.tsx    # minimal markdown + code blocks
        ├── hooks/
        │   └── useAgent.ts     # WS client, reconnect, event→state, actions
        └── components/
            └── Chat.tsx        # Turn, Thinking, SearchCard, EventStream
```

---

## 4. Setup & run

### Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # then edit .env and paste your OpenAI key
uvicorn main:app --port 8000
```

Verify: open `http://localhost:8000/healthz` → `{"ok": true, "hasKey": true}`.

> Tip: omit `--reload` on Windows/OneDrive to avoid noisy file-watch reloads.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the printed URL (default `http://localhost:5173`). The status dot turns
green when connected. Point at another backend with
`VITE_WS_URL=ws://host:8000/ws npm run dev`.

---

## 5. Things to try

- "Explain async/await with a code example" → streamed answer + code block.
- "What are the latest developments in the AG-UI protocol?" → watch the
  **web search** card run, then the answer stream in.
- Start a long answer, then click **■** → generation halts immediately
  (server-side cancel).
- While it's generating, type "keep it under 3 sentences" and press Enter →
  the instruction is **injected** into the running task (you'll see `inject.ack`).
- **↻ Regenerate** re-runs the last user message.

---

## 6. Notes, limits, and honesty

- **Real SDK, untested-here:** the code uses `ag_ui.core` and OpenAI, which
  couldn't be installed/run in the authoring environment (no network). The agent
  logic (streaming, tool loop, server-side cancel, injection) was verified with
  mocks. First real run may surface a version-specific event field name — if an
  import in `events.py` fails, the error names the exact symbol to fix.
- **`model` via `forwardedProps`:** the official `RunAgentInput` has no standard
  `model` field, so the chosen model is passed in `forwardedProps.model` and read
  server-side from `run_input.forwarded_props`.
- **Thinking visibility:** standard OpenAI models don't expose raw chain-of-thought.
  The "thinking" block shows the agent's pre-answer/tool-planning phase as a
  distinct state. For models with real reasoning tokens, you'd map those to the
  AG-UI reasoning events instead.
- **web_search:** uses DuckDuckGo (no key). If `duckduckgo-search` isn't installed
  it falls back to DDG's JSON endpoint; if both fail it returns an explicit
  "no results" rather than fabricating.
- Demo-grade: CORS is open, no auth. Lock these down before deploying.
```
