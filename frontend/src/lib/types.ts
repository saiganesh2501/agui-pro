// AG-UI event types — these mirror the wire format produced by the official
// ag_ui.core Pydantic models (camelCase via by_alias=True). The backend emits
// the real SDK events; this is the typed shape the client consumes.

export enum EventType {
  RUN_STARTED = "RUN_STARTED",
  RUN_FINISHED = "RUN_FINISHED",
  RUN_ERROR = "RUN_ERROR",
  STEP_STARTED = "STEP_STARTED",
  STEP_FINISHED = "STEP_FINISHED",
  TEXT_MESSAGE_START = "TEXT_MESSAGE_START",
  TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT",
  TEXT_MESSAGE_END = "TEXT_MESSAGE_END",
  TOOL_CALL_START = "TOOL_CALL_START",
  TOOL_CALL_ARGS = "TOOL_CALL_ARGS",
  TOOL_CALL_END = "TOOL_CALL_END",
  TOOL_CALL_RESULT = "TOOL_CALL_RESULT",
  STATE_SNAPSHOT = "STATE_SNAPSHOT",
  STATE_DELTA = "STATE_DELTA",
  CUSTOM = "CUSTOM",
  RAW = "RAW",
}

export interface JsonPatchOp {
  op: "add" | "remove" | "replace" | "move" | "copy" | "test";
  path: string;
  value?: unknown;
  from?: string;
}

export type AgUiEvent = {
  type: EventType;
  timestamp?: number;
  threadId?: string;
  runId?: string;
  message?: string;
  code?: string;
  stepName?: string;
  messageId?: string;
  role?: string;
  delta?: string | JsonPatchOp[];
  toolCallId?: string;
  toolCallName?: string;
  parentMessageId?: string;
  content?: string;
  snapshot?: Record<string, unknown>;
  name?: string;
  value?: unknown;
  event?: unknown;
  source?: string;
};

export type EventCategory =
  | "lifecycle" | "text" | "tool" | "state" | "thinking" | "custom" | "raw";

export function categoryOf(type: EventType, name?: string): EventCategory {
  if (type === EventType.CUSTOM && name?.startsWith("thinking")) return "thinking";
  if (type.startsWith("RUN")) return "lifecycle";
  if (type.startsWith("STEP")) return "thinking";
  if (type.startsWith("TEXT_MESSAGE")) return "text";
  if (type.startsWith("TOOL_CALL")) return "tool";
  if (type.startsWith("STATE")) return "state";
  if (type === EventType.CUSTOM) return "custom";
  return "raw";
}

// Outbound control messages (client -> server)
export type ClientMessage =
  | { type: "run"; input: RunAgentInput }
  | { type: "stop" }
  | { type: "inject"; text: string };

export interface RunAgentInput {
  threadId: string;
  runId: string;
  messages: { id: string; role: string; content: string }[];
  state: Record<string, unknown>;
  tools: unknown[];
  context: unknown[];
  forwardedProps: Record<string, unknown>;
}

export const MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"] as const;

// ----- derived UI types -----
export interface ChatTurn {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming: boolean;
  thinking: string;        // accumulated thinking text
  thinkingActive: boolean; // currently in a thinking phase
  searches: SearchView[];
}

export interface SearchView {
  id: string;
  query: string;
  status: "searching" | "done";
  results: { title: string; url: string; snippet: string }[];
  _raw?: string; // internal: accumulates streamed arg fragments
}
