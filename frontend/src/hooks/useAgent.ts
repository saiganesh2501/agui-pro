import { useCallback, useEffect, useRef, useState } from "react";
import {
  AgUiEvent, ChatTurn, ClientMessage, EventType, RunAgentInput, SearchView,
} from "../lib/types";

const WS_URL = import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws";
export type ConnStatus = "connecting" | "open" | "closed";

export function useAgent(model: string) {
  const wsRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<ConnStatus>("connecting");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [events, setEvents] = useState<AgUiEvent[]>([]);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const turnsRef = useRef<ChatTurn[]>([]);
  turnsRef.current = turns;
  const modelRef = useRef(model);
  modelRef.current = model;

  // ---- connection with auto-reconnect ----
  useEffect(() => {
    let closed = false;
    let retry: ReturnType<typeof setTimeout>;
    let attempts = 0;

    const connect = () => {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      setStatus("connecting");

      ws.onopen = () => {
        attempts = 0;
        setStatus("open");
        setError(null);
      };
      ws.onclose = () => {
        setStatus("closed");
        setGenerating(false);
        if (!closed) {
          // exponential backoff capped at 8s
          const delay = Math.min(1000 * 2 ** attempts, 8000);
          attempts += 1;
          retry = setTimeout(connect, delay);
        }
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (m) => {
        try {
          handleEvent(JSON.parse(m.data) as AgUiEvent);
        } catch {
          /* ignore malformed frames */
        }
      };
    };

    connect();
    return () => {
      closed = true;
      clearTimeout(retry);
      wsRef.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const send = useCallback((msg: ClientMessage) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
  }, []);

  // ---- event → state mapping ----
  const handleEvent = useCallback((ev: AgUiEvent) => {
    setEvents((prev) => [...prev, ev]);

    switch (ev.type) {
      case EventType.RUN_STARTED:
        setGenerating(true);
        setError(null);
        break;

      case EventType.RUN_FINISHED:
        setGenerating(false);
        setTurns((t) => finalizeStreaming(t));
        break;

      case EventType.RUN_ERROR:
        setGenerating(false);
        setTurns((t) => finalizeStreaming(t));
        if (ev.code !== "STOPPED") setError(ev.message ?? "Run error");
        break;

      // thinking phase (STEP_STARTED "thinking" / CUSTOM thinking.delta)
      case EventType.STEP_STARTED:
        if (ev.stepName === "thinking") {
          setTurns((t) => withAssistant(t, (a) => ({ ...a, thinkingActive: true })));
        }
        break;
      case EventType.STEP_FINISHED:
        if (ev.stepName === "thinking") {
          setTurns((t) => withAssistant(t, (a) => ({ ...a, thinkingActive: false })));
        }
        break;

      // streamed assistant answer
      case EventType.TEXT_MESSAGE_START:
        setTurns((t) => ensureAssistant(t, ev.messageId!));
        break;
      case EventType.TEXT_MESSAGE_CONTENT:
        setTurns((t) =>
          withAssistant(t, (a) => ({ ...a, content: a.content + ((ev.delta as string) ?? "") }))
        );
        break;
      case EventType.TEXT_MESSAGE_END:
        setTurns((t) => withAssistant(t, (a) => ({ ...a, streaming: false })));
        break;

      // web search tool surfaced in UI
      case EventType.TOOL_CALL_START:
        if (ev.toolCallName === "web_search") {
          setTurns((t) =>
            withAssistant(t, (a) => ({
              ...a,
              searches: [
                ...a.searches,
                { id: ev.toolCallId!, query: "", status: "searching", results: [] },
              ],
            }))
          );
        }
        break;
      case EventType.TOOL_CALL_ARGS:
        // accumulate raw arg fragments; parse out the query once it's complete
        setTurns((t) =>
          withAssistant(t, (a) => ({
            ...a,
            searches: a.searches.map((sv) =>
              sv.id === ev.toolCallId
                ? (() => {
                    const raw = (sv as any)._raw ? (sv as any)._raw + ((ev.delta as string) ?? "") : ((ev.delta as string) ?? "");
                    const m = raw.match(/"query"\s*:\s*"([^"]*)"/);
                    return { ...sv, _raw: raw, query: m ? m[1] : sv.query } as SearchView;
                  })()
                : sv
            ),
          }))
        );
        break;
      case EventType.TOOL_CALL_RESULT: {
        const parsed = safeParse(ev.content);
        setTurns((t) =>
          withAssistant(t, (a) => ({
            ...a,
            searches: a.searches.map((sv) =>
              sv.id === ev.toolCallId
                ? {
                    ...sv,
                    status: "done",
                    query: parsed?.query ?? sv.query,
                    results: parsed?.results ?? [],
                  }
                : sv
            ),
          }))
        );
        break;
      }

      case EventType.CUSTOM:
        if (ev.name === "thinking.delta") {
          const v = ev.value as { delta: string };
          setTurns((t) =>
            withAssistant(t, (a) => ({ ...a, thinking: a.thinking + (v.delta ?? "") }))
          );
        }
        break;
    }
  }, []);

  // ---- actions ----
  const runWith = useCallback(
    (history: ChatTurn[]) => {
      const messages = history.map((t) => ({ id: t.id, role: t.role, content: t.content }));
      const input: RunAgentInput = {
        threadId: "thread_main",
        runId: `run_${Date.now()}`,
        messages,
        state: {},
        tools: [],
        context: [],
        forwardedProps: { model: modelRef.current },
      };
      send({ type: "run", input });
    },
    [send]
  );

  const sendMessage = useCallback(
    (text: string) => {
      const userTurn: ChatTurn = {
        id: `u_${Date.now()}`, role: "user", content: text,
        streaming: false, thinking: "", thinkingActive: false, searches: [],
      };
      const history = [...turnsRef.current, userTurn];
      setTurns(history);
      runWith(history);
    },
    [runWith]
  );

  const stop = useCallback(() => send({ type: "stop" }), [send]);

  // Inject extra instructions into the ongoing task (or, if idle, send as a
  // normal new message). The server folds it into the active run.
  const inject = useCallback(
  (text: string) => {
    if (generating) {
      // stop the current run and restart with the instruction added
      send({ type: "stop" });
      const noteTurn: ChatTurn = {
        id: `u_${Date.now()}`, role: "user", content: text,
        streaming: false, thinking: "", thinkingActive: false, searches: [],
      };
      const history = [...turnsRef.current, noteTurn];
      setTurns(history);
      // small delay so the stop is processed before the new run
      setTimeout(() => runWith(history), 150);
    } else {
      sendMessage(text);
    }
  },
  [generating, send, sendMessage, runWith]
  );

  const regenerate = useCallback(() => {
    const t = turnsRef.current;
    const lastUserIdx = [...t].reverse().findIndex((x) => x.role === "user");
    if (lastUserIdx === -1) return;
    const idx = t.length - 1 - lastUserIdx;
    const trimmed = t.slice(0, idx + 1);
    setTurns(trimmed);
    runWith(trimmed);
  }, [runWith]);

  const newChat = useCallback(() => {
    setTurns([]);
    setError(null);
  }, []);

  const clearEvents = useCallback(() => setEvents([]), []);

  return {
    status, turns, events, generating, error,
    sendMessage, stop, inject, regenerate, newChat, clearEvents,
  };
}

// ---------- pure helpers ----------
function ensureAssistant(turns: ChatTurn[], id: string): ChatTurn[] {
  const last = turns[turns.length - 1];
  if (last && last.role === "assistant" && last.streaming) {
    // already have a streaming assistant turn; keep it (id may differ per message)
    return turns;
  }
  return [
    ...turns,
    { id, role: "assistant", content: "", streaming: true, thinking: "", thinkingActive: false, searches: [] },
  ];
}

function withAssistant(turns: ChatTurn[], fn: (a: ChatTurn) => ChatTurn): ChatTurn[] {
  // apply to the last assistant turn, creating one if none is streaming
  let list = turns;
  const lastIdx = [...list].map((t) => t.role).lastIndexOf("assistant");
  if (lastIdx === -1 || !list[lastIdx].streaming) {
    list = [
      ...list,
      { id: `a_${Date.now()}`, role: "assistant", content: "", streaming: true, thinking: "", thinkingActive: false, searches: [] },
    ];
    return list.map((t, i) => (i === list.length - 1 ? fn(t) : t));
  }
  return list.map((t, i) => (i === lastIdx ? fn(t) : t));
}

function finalizeStreaming(turns: ChatTurn[]): ChatTurn[] {
  return turns.map((t) => ({ ...t, streaming: false, thinkingActive: false }));
}

function safeParse(s?: string): any {
  if (!s) return null;
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}
