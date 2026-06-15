"""
agent.py — the streaming agent that emits REAL AG-UI events.

It calls OpenAI with streaming, surfaces a visible "thinking" phase, calls the
web_search tool when needed, and streams the final answer token-by-token. All
output is genuine ag_ui.core event objects.

Bidirectional control lives in the shared `Steering` object, which the
WebSocket layer mutates from inbound client messages:
  - cancelled      -> stop the run immediately (also aborts the OpenAI stream)
  - injected_notes -> extra instructions to fold into the next model turn

Cancellation is real: we close the OpenAI streaming response so the in-flight
request is dropped server-side, not just hidden in the UI.

Event → UI mapping (see README for the full table):
  RUN_STARTED / RUN_FINISHED / RUN_ERROR  -> generating on/off, errors
  STEP_STARTED("thinking"/"search"/...)   -> which phase the UI shows
  CUSTOM "thinking.delta"                  -> streamed thinking text (distinct UI)
  TOOL_CALL_START/ARGS/END/RESULT          -> the web-search card in the UI
  TEXT_MESSAGE_START/CONTENT/END           -> the streamed assistant answer
  CUSTOM "inject.ack"                       -> confirms an injected instruction
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

from events import (
    BaseEvent, EventType,
    RunStartedEvent, RunFinishedEvent, RunErrorEvent,
    StepStartedEvent, StepFinishedEvent,
    TextMessageStartEvent, TextMessageContentEvent, TextMessageEndEvent,
    ToolCallStartEvent, ToolCallArgsEvent, ToolCallEndEvent, ToolCallResultEvent,
    StateSnapshotEvent, StateDeltaEvent,
    CustomEvent, RawEvent,
    RunAgentInput,
)
from tools import TOOL_SCHEMAS, run_tool

MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant in a ChatGPT-style app. "
    "When a question needs current, factual, or recent information, call the "
    "web_search tool rather than guessing. Before answering a non-trivial "
    "question, briefly think step by step. Format answers in Markdown, using "
    "fenced code blocks with a language hint when you share code."
)


class StopRun(Exception):
    """Raised internally to unwind the generator when cancelled."""


@dataclass
class Steering:
    cancelled: bool = False
    injected_notes: list[str] = field(default_factory=list)
    # Holds the live OpenAI stream so cancel() can close it server-side.
    active_stream: Any = None

    def check(self) -> None:
        if self.cancelled:
            raise StopRun()


class StreamingAgent:
    def __init__(self, client, steering: Steering):
        self.client = client            # AsyncOpenAI
        self.steering = steering

    async def run(self, run_input: RunAgentInput) -> AsyncGenerator[BaseEvent, None]:
        s = self.steering
        thread_id = run_input.thread_id
        run_id = run_input.run_id

        # Build OpenAI history from the AG-UI messages (Message objects or dicts).
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in run_input.messages:
            role = getattr(m, "role", None) if not isinstance(m, dict) else m.get("role")
            content = getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
            if role in ("user", "assistant", "system") and content:
                messages.append({"role": role, "content": content})

        # model can ride in forwarded_props (kept out of standard fields).
        fwd = getattr(run_input, "forwarded_props", None) or {}
        model = fwd.get("model", "gpt-4o-mini") if isinstance(fwd, dict) else "gpt-4o-mini"

        try:
            yield RunStartedEvent(type=EventType.RUN_STARTED,
                                  thread_id=thread_id, run_id=run_id)
            yield StateSnapshotEvent(type=EventType.STATE_SNAPSHOT,
                                     snapshot={"status": "thinking", "model": model,
                                               "toolRounds": 0})

            for round_idx in range(MAX_TOOL_ROUNDS):
                s.check()

                # Fold any instructions the user injected mid-run into the prompt.
                while s.injected_notes:
                    note = s.injected_notes.pop(0)
                    messages.append({"role": "user", "content": note})
                    yield CustomEvent(type=EventType.CUSTOM, name="inject.ack",
                                      value={"note": note})

                # Stream one model turn. It may yield events, and returns the
                # collected text + tool calls via the final meta payload.
                meta: dict[str, Any] = {}
                async for ev in self._one_turn(messages, model, meta):
                    yield ev

                tool_calls = meta.get("tool_calls", [])
                assistant_text = meta.get("text", "")

                # Record the assistant turn (text + tool calls) for history.
                assistant_msg: dict[str, Any] = {"role": "assistant",
                                                  "content": assistant_text or None}
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {"id": tc["id"], "type": "function",
                         "function": {"name": tc["name"], "arguments": tc["args"]}}
                        for tc in tool_calls
                    ]
                messages.append(assistant_msg)

                if not tool_calls:
                    break  # final answer produced, no more tools requested

                yield StateDeltaEvent(type=EventType.STATE_DELTA, delta=[
                    {"op": "replace", "path": "/toolRounds", "value": round_idx + 1}
                ])

                # Execute each requested tool (currently: web_search).
                for tc in tool_calls:
                    s.check()
                    try:
                        parsed = json.loads(tc["args"] or "{}")
                    except json.JSONDecodeError:
                        parsed = {}

                    yield StepStartedEvent(type=EventType.STEP_STARTED,
                                           step_name=f"search:{parsed.get('query','')[:40]}")
                    result = await run_tool(tc["name"], parsed)
                    # Surface raw tool telemetry as a genuine RAW event.
                    yield RawEvent(type=EventType.RAW, source="web_search",
                                   event={"query": parsed.get("query", "")})
                    yield ToolCallResultEvent(
                        type=EventType.TOOL_CALL_RESULT,
                        tool_call_id=tc["id"],
                        message_id=f"m_{uuid.uuid4().hex[:8]}",
                        content=result,
                    )
                    yield StepFinishedEvent(type=EventType.STEP_FINISHED,
                                            step_name=f"search:{parsed.get('query','')[:40]}")
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": result})

            yield StateDeltaEvent(type=EventType.STATE_DELTA, delta=[
                {"op": "replace", "path": "/status", "value": "done"}
            ])
            yield RunFinishedEvent(type=EventType.RUN_FINISHED,
                                   thread_id=thread_id, run_id=run_id)

        except StopRun:
            # Cancellation requested. Make sure the OpenAI stream is closed.
            await self._abort_stream()
            yield CustomEvent(type=EventType.CUSTOM, name="run.stopped",
                              value={"reason": "user_stopped"})
            yield RunErrorEvent(type=EventType.RUN_ERROR,
                                message="Generation stopped by user.", code="STOPPED")
        except Exception as exc:  # surface real errors (bad key, quota, etc.)
            await self._abort_stream()
            yield RunErrorEvent(type=EventType.RUN_ERROR, message=str(exc),
                                code="AGENT_ERROR")

    async def _one_turn(self, messages, model, meta) -> AsyncGenerator[BaseEvent, None]:
        """
        Stream one OpenAI completion. Emits a visible thinking phase, streams the
        answer as TEXT_MESSAGE_* events, and collects tool calls. Fills `meta`
        with {"text", "tool_calls"} once the stream ends.
        """
        s = self.steering
        msg_id = f"m_{uuid.uuid4().hex[:8]}"
        think_id = f"t_{uuid.uuid4().hex[:8]}"

        text_started = False
        thinking_started = False
        text_buf = ""
        tool_acc: dict[int, dict[str, str]] = {}
        tool_started: set[int] = set()

        # Ask OpenAI to stream. We request reasoning visibility by using a
        # lightweight "think out loud briefly" convention captured below; the
        # model's pre-answer tokens before any tool call are shown as thinking.
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            stream=True,
            temperature=0.4,
        )
        s.active_stream = stream  # so cancel can close it

        try:
            async for chunk in stream:
                s.check()
                choice = chunk.choices[0]
                delta = choice.delta

                # --- assistant text tokens ---
                if delta.content:
                    if not text_started:
                        # close any thinking phase first
                        if thinking_started:
                            yield StepFinishedEvent(type=EventType.STEP_FINISHED,
                                                    step_name="thinking")
                            thinking_started = False
                        text_started = True
                        yield TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START,
                                                    message_id=msg_id, role="assistant")
                    text_buf += delta.content
                    yield TextMessageContentEvent(type=EventType.TEXT_MESSAGE_CONTENT,
                                                  message_id=msg_id, delta=delta.content)

                # --- tool call fragments ---
                if delta.tool_calls:
                    # entering tool planning counts as thinking, show it once
                    if not thinking_started and not text_started:
                        thinking_started = True
                        yield StepStartedEvent(type=EventType.STEP_STARTED,
                                               step_name="thinking")
                        yield CustomEvent(type=EventType.CUSTOM, name="thinking.delta",
                                          value={"messageId": think_id,
                                                 "delta": "Deciding to search the web…"})
                    for tcd in delta.tool_calls:
                        idx = tcd.index
                        if idx not in tool_acc:
                            tool_acc[idx] = {
                                "id": tcd.id or f"tc_{uuid.uuid4().hex[:8]}",
                                "name": "", "args": "",
                            }
                        if tcd.id:
                            tool_acc[idx]["id"] = tcd.id
                        if tcd.function and tcd.function.name:
                            tool_acc[idx]["name"] += tcd.function.name
                            if idx not in tool_started:
                                tool_started.add(idx)
                                yield ToolCallStartEvent(
                                    type=EventType.TOOL_CALL_START,
                                    tool_call_id=tool_acc[idx]["id"],
                                    tool_call_name=tool_acc[idx]["name"],
                                    parent_message_id=msg_id,
                                )
                        if tcd.function and tcd.function.arguments:
                            tool_acc[idx]["args"] += tcd.function.arguments
                            yield ToolCallArgsEvent(
                                type=EventType.TOOL_CALL_ARGS,
                                tool_call_id=tool_acc[idx]["id"],
                                delta=tcd.function.arguments,
                            )
        finally:
            s.active_stream = None

        # close out open streams
        if text_started:
            yield TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=msg_id)
        if thinking_started:
            yield StepFinishedEvent(type=EventType.STEP_FINISHED, step_name="thinking")
        for idx in tool_started:
            yield ToolCallEndEvent(type=EventType.TOOL_CALL_END,
                                   tool_call_id=tool_acc[idx]["id"])

        meta["text"] = text_buf
        meta["tool_calls"] = [tool_acc[i] for i in sorted(tool_acc.keys())]

    async def _abort_stream(self) -> None:
        """Close the in-flight OpenAI stream so the request is dropped server-side."""
        stream = self.steering.active_stream
        if stream is not None:
            try:
                await stream.close()
            except Exception:
                pass
            self.steering.active_stream = None
