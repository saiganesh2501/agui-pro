"""
agent.py — streaming agent emitting real AG-UI events, NOW WITH LANGFUSE TRACING + PROMPTS + EVALUATION.

What changed vs. the base version (everything else is identical behavior):
  1. The OpenAI client is the Langfuse-wrapped one (from langfuse.openai import
     AsyncOpenAI). Every chat.completions.create call is auto-captured as a
     Langfuse "generation" with model, prompt, completion, tokens, latency, cost
     — including streamed responses.
  2. Each agent run is wrapped in a Langfuse trace via the wrapped client,
     so all generations + the web-search span nest under one trace per user message.
  3. System prompt is fetched from Langfuse at runtime (with fallback).
  4. After each run, LLM-as-judge evaluation runs and scores are attached to trace.

Tracing is OPTIONAL: if Langfuse isn't configured, observability.is_enabled()
is False and the agent runs exactly as before (no tracing, no errors).
"""
from __future__ import annotations

import asyncio
import json
import logging
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
import observability as obs
from prompt_manager import get_prompt_manager
from observability import get_observability

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6


class StopRun(Exception):
    """Raised internally to unwind the generator when cancelled."""


@dataclass
class Steering:
    cancelled: bool = False
    injected_notes: list[str] = field(default_factory=list)
    active_stream: Any = None
    # Set when a run starts, so main.py can attach feedback scores to the trace.
    trace_id: Optional[str] = None

    def check(self) -> None:
        if self.cancelled:
            raise StopRun()


class StreamingAgent:
    def __init__(self, client, steering: Steering):
        self.client = client            # Langfuse-wrapped AsyncOpenAI
        self.steering = steering
        # Initialize managers
        self.prompt_manager = get_prompt_manager()
        self.obs = get_observability()

    async def run(self, run_input: RunAgentInput) -> AsyncGenerator[BaseEvent, None]:
        """
        Run the agent. The Langfuse-wrapped OpenAI client (used in main.py)
        auto-creates traces + generations for every LLM call.
        
        After run completes, evaluation runs and scores are attached to Langfuse.
        """
        trace_id = self.steering.trace_id or f"trace_{uuid.uuid4().hex[:8]}"
        used_search = False
        final_response = ""
        # Safely extract the last user message (handles dicts AND Pydantic objects)
        user_query = _last_user_text(run_input)

        try:
            async for ev in self._run_impl(run_input):
                yield ev

                # Track if web search was used
                if isinstance(ev, ToolCallEndEvent) and getattr(ev, "tool_call_name", None) == "web_search":
                    used_search = True

                # Collect final response text
                if isinstance(ev, TextMessageContentEvent):
                    final_response += ev.delta

        finally:
            # Flush traces to Langfuse
            if obs.is_enabled():
                obs.flush()

            # Use the REAL Langfuse trace id captured during the run (set in
            # _one_turn). Falls back to the placeholder only if none was captured.
            eval_trace_id = self.steering.trace_id or trace_id

            # Evaluate the run as a BACKGROUND task so it never blocks the
            # response or breaks the run if an eval call hangs/errors.
            if final_response.strip() and self.obs.is_enabled():
                asyncio.create_task(
                    self._safe_evaluate_run(
                        trace_id=eval_trace_id,
                        query=user_query,
                        response=final_response,
                        used_search=used_search,
                    )
                )

    async def _safe_evaluate_run(self, trace_id, query, response, used_search):
        """Wrapper that swallows all evaluation errors so they never surface."""
        try:
            await self._evaluate_run(
                trace_id=trace_id,
                query=query,
                response=response,
                used_search=used_search,
            )
        except Exception as e:
            logger.warning(f"Background evaluation failed: {e}")

    async def _evaluate_run(
        self,
        trace_id: str,
        query: str,
        response: str,
        used_search: bool,
    ):
        """
        Evaluate the agent's response and attach scores to Langfuse trace.
        
        Runs three evaluations:
        1. Quality: is the response well-formatted?
        2. Relevance: does it answer the question?
        3. Correctness: are the facts accurate?
        """
        if not self.obs.is_enabled():
            logger.debug("Evaluation disabled")
            return

        logger.info(f"Evaluating trace {trace_id[:8]}...")

        try:
            # Evaluation 1: Quality (always run)
            await self.obs.evaluate_and_score(
                trace_id=trace_id,
                user_query=query,
                response=response,
                eval_type="quality",
            )

            # Evaluation 2: Relevance (always run)
            await self.obs.evaluate_and_score(
                trace_id=trace_id,
                user_query=query,
                response=response,
                eval_type="relevance",
            )

            # Evaluation 3: Correctness (if search was used)
            if used_search:
                await self.obs.evaluate_and_score(
                    trace_id=trace_id,
                    user_query=query,
                    response=response,
                    eval_type="correctness",
                )

            logger.info(f"✓ Evaluations complete for {trace_id[:8]}...")

        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")

    async def _run_impl(self, run_input: RunAgentInput) -> AsyncGenerator[BaseEvent, None]:
        s = self.steering
        thread_id = run_input.thread_id
        run_id = run_input.run_id

        # ┌─── NEW: Fetch system prompt from Langfuse ───┐
        system_prompt = self.obs.get_system_prompt()
        # └────────────────────────────────────────────────┘

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for m in run_input.messages:
            role = getattr(m, "role", None) if not isinstance(m, dict) else m.get("role")
            content = getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
            if role in ("user", "assistant", "system") and content:
                messages.append({"role": role, "content": content})

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

                while s.injected_notes:
                    note = s.injected_notes.pop(0)
                    messages.append({"role": "user", "content": note})
                    yield CustomEvent(type=EventType.CUSTOM, name="inject.ack",
                                      value={"note": note})

                meta: dict[str, Any] = {}
                async for ev in self._one_turn(messages, model, meta):
                    yield ev

                tool_calls = meta.get("tool_calls", [])
                assistant_text = meta.get("text", "")

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
                    break

                yield StateDeltaEvent(type=EventType.STATE_DELTA, delta=[
                    {"op": "replace", "path": "/toolRounds", "value": round_idx + 1}
                ])

                for tc in tool_calls:
                    s.check()
                    try:
                        parsed = json.loads(tc["args"] or "{}")
                    except json.JSONDecodeError:
                        parsed = {}

                    query = parsed.get("query", "")
                    yield StepStartedEvent(type=EventType.STEP_STARTED,
                                           step_name=f"search:{query[:40]}")

                    # --- web search recorded as its own Langfuse span ---
                    result = await self._tool_with_span(tc["name"], parsed)

                    yield RawEvent(type=EventType.RAW, source="web_search",
                                   event={"query": query})
                    yield ToolCallResultEvent(
                        type=EventType.TOOL_CALL_RESULT,
                        tool_call_id=tc["id"],
                        message_id=f"m_{uuid.uuid4().hex[:8]}",
                        content=result,
                    )
                    yield StepFinishedEvent(type=EventType.STEP_FINISHED,
                                            step_name=f"search:{query[:40]}")
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": result})

            yield StateDeltaEvent(type=EventType.STATE_DELTA, delta=[
                {"op": "replace", "path": "/status", "value": "done"}
            ])
            yield RunFinishedEvent(type=EventType.RUN_FINISHED,
                                   thread_id=thread_id, run_id=run_id)

        except StopRun:
            await self._abort_stream()
            yield CustomEvent(type=EventType.CUSTOM, name="run.stopped",
                              value={"reason": "user_stopped"})
            yield RunErrorEvent(type=EventType.RUN_ERROR,
                                message="Generation stopped by user.", code="STOPPED")
        except Exception as exc:
            await self._abort_stream()
            yield RunErrorEvent(type=EventType.RUN_ERROR, message=str(exc),
                                code="AGENT_ERROR")

    async def _tool_with_span(self, name: str, args: dict[str, Any]) -> str:
        """
        Run a tool. The Langfuse-wrapped client auto-traces tool execution.
        """
        return await run_tool(name, args)

    async def _one_turn(self, messages, model, meta) -> AsyncGenerator[BaseEvent, None]:
        s = self.steering
        msg_id = f"m_{uuid.uuid4().hex[:8]}"
        think_id = f"t_{uuid.uuid4().hex[:8]}"

        text_started = False
        thinking_started = False
        text_buf = ""
        tool_acc: dict[int, dict[str, str]] = {}
        tool_started: set[int] = set()

        # The Langfuse-wrapped client auto-creates a "generation" for this call.
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            stream=True,
            temperature=0.4,
        )
        s.active_stream = stream

        # Capture the REAL Langfuse trace ID (so eval scores attach to the
        # actual trace shown in the dashboard, not a made-up id). Only set it
        # once per run, on the first LLM call.
        if obs.is_enabled() and s.trace_id is None:
            try:
                real_id = obs.get_current_trace_id()
                if real_id:
                    s.trace_id = real_id
            except Exception:
                pass

        try:
            async for chunk in stream:
                s.check()
                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    if not text_started:
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

                if delta.tool_calls:
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
        stream = self.steering.active_stream
        if stream is not None:
            try:
                await stream.close()
            except Exception:
                pass
            self.steering.active_stream = None


def _last_user_text(run_input: RunAgentInput) -> str:
    for m in reversed(run_input.messages):
        role = getattr(m, "role", None) if not isinstance(m, dict) else m.get("role")
        if role == "user":
            return getattr(m, "content", None) if not isinstance(m, dict) else m.get("content", "")
    return ""