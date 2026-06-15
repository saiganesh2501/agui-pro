"""
main.py — FastAPI WebSocket server wiring the streaming agent to the browser.

One socket carries both directions:

  client -> server (JSON)
    { "type": "run",    "input": { ...RunAgentInput (camelCase)... } }
    { "type": "stop" }                      # cancel the current run immediately
    { "type": "inject", "text": "..." }     # add instructions to the ongoing task
  server -> client (JSON)
    Any official ag_ui.core event, serialized to camelCase via by_alias=True.

Cancellation is real: "stop" flips Steering.cancelled AND closes the live
OpenAI stream, so the in-flight request is dropped server-side.

Run:
    uvicorn main:app --reload --port 8000   (needs OPENAI_API_KEY in env / .env)
"""
from __future__ import annotations

import asyncio
import contextlib
import os

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

from agent import StreamingAgent, Steering, StopRun
from events import BaseEvent, CustomEvent, RunErrorEvent, RunAgentInput

load_dotenv()

app = FastAPI(title="AG-UI Streaming Chat Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # demo; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


def serialize(event: BaseEvent) -> dict:
    """Official AG-UI events serialize to camelCase via Pydantic aliases."""
    return event.model_dump(by_alias=True, mode="json", exclude_none=True)


@app.get("/healthz")
async def healthz():
    return {"ok": True, "hasKey": bool(os.getenv("OPENAI_API_KEY"))}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    api_key = os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key) if api_key else None
    if client is None:
        await ws.send_json(serialize(RunErrorEvent(
            type="RUN_ERROR",
            message="OPENAI_API_KEY is not set on the server. Add it to backend/.env and restart.",
            code="NO_API_KEY",
        )))

    steering = Steering()
    agent_task: asyncio.Task | None = None
    send_lock = asyncio.Lock()

    async def send_event(event: BaseEvent) -> None:
        async with send_lock:
            with contextlib.suppress(Exception):
                await ws.send_json(serialize(event))

    async def drive(run_input: RunAgentInput) -> None:
        agent = StreamingAgent(client, steering)
        try:
            async for event in agent.run(run_input):
                await send_event(event)
        except StopRun:
            pass

    async def cancel_current() -> None:
        """Stop the running agent cleanly and close its OpenAI stream."""
        nonlocal agent_task
        if agent_task and not agent_task.done():
            steering.cancelled = True
            # proactively close the in-flight OpenAI stream
            stream = steering.active_stream
            if stream is not None:
                with contextlib.suppress(Exception):
                    await stream.close()
            with contextlib.suppress(Exception):
                await agent_task

    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")

            if kind == "run":
                if client is None:
                    await send_event(RunErrorEvent(
                        type="RUN_ERROR",
                        message="No OpenAI API key configured on the server.",
                        code="NO_API_KEY"))
                    continue
                await cancel_current()               # stop any previous run
                steering = Steering()                # fresh control state
                run_input = RunAgentInput(**msg["input"])
                agent_task = asyncio.create_task(drive(run_input))

            elif kind == "stop":
                await cancel_current()
                await send_event(CustomEvent(type="CUSTOM", name="control.ack",
                                             value={"state": "stopped"}))

            elif kind == "inject":
                text = str(msg.get("text", "")).strip()
                if text:
                    steering.injected_notes.append(text)
                    await send_event(CustomEvent(type="CUSTOM", name="control.ack",
                                     value={"state": "injected", "text": text}))
                    # NEW: stop the current generation so the note is applied immediately
                if agent_task and not agent_task.done():
                    await cancel_current()

    except WebSocketDisconnect:
        await cancel_current()
