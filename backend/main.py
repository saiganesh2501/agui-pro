"""
main.py — FastAPI WebSocket server, NOW WITH LANGFUSE.

Two changes vs. the base version:
  1. The OpenAI client is imported from langfuse.openai (the wrapped drop-in).
     If Langfuse isn't installed/configured, we fall back to the plain client.
  2. A new control message {type:"feedback", score:1|0, comment} attaches a
     user rating to the current run's Langfuse trace (via Steering.trace_id).

Everything else (run/stop/inject, cancellation) is unchanged.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from agent import StreamingAgent, Steering, StopRun
from events import BaseEvent, CustomEvent, RunErrorEvent, RunAgentInput
from observability import init_observability
import observability as obs

load_dotenv()

# Configure logging so all logger.info/error calls actually print to the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# Create OpenAI client (plain version first, will be wrapped if Langfuse enabled)
# ════════════════════════════════════════════════════════════════════════════════
from openai import AsyncOpenAI
api_key = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=api_key) if api_key else None

# ════════════════════════════════════════════════════════════════════════════════
# Initialize observability FIRST (prompts + evaluation + Langfuse tracing)
# This wraps the OpenAI client with Langfuse if enabled
# ════════════════════════════════════════════════════════════════════════════════
init_observability(openai_client)
logger.info(f"✓ Observability initialized: Langfuse={'enabled' if obs.is_enabled() else 'disabled'}")

# Now use Langfuse-wrapped client if available
if obs.is_enabled():
    try:
        from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI
        openai_client = LangfuseAsyncOpenAI(api_key=api_key)
        logger.info("✓ Using Langfuse-wrapped OpenAI client")
    except Exception as e:
        logger.warning(f"Failed to use Langfuse-wrapped client: {e}. Using plain client.")
# ════════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="AG-UI Streaming Chat Backend (Langfuse)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def serialize(event: BaseEvent) -> dict:
    return event.model_dump(by_alias=True, mode="json", exclude_none=True)


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "hasKey": bool(api_key),
        "langfuse": obs.is_enabled(),
        "prompts": "enabled" if obs.is_enabled() else "disabled",
        "evaluation": "enabled" if obs.is_enabled() else "disabled",
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info(f"Client connected")

    if openai_client is None:
        await ws.send_json(serialize(RunErrorEvent(
            type="RUN_ERROR",
            message="OPENAI_API_KEY is not set on the server. Add it to backend/.env and restart.",
            code="NO_API_KEY",
        )))
        await ws.close()
        return

    steering = Steering()
    agent_task: asyncio.Task | None = None
    send_lock = asyncio.Lock()

    async def send_event(event: BaseEvent) -> None:
        async with send_lock:
            with contextlib.suppress(Exception):
                await ws.send_json(serialize(event))

    async def drive(run_input: RunAgentInput) -> None:
        agent = StreamingAgent(openai_client, steering)
        try:
            async for event in agent.run(run_input):
                await send_event(event)
        except StopRun:
            pass
        except Exception as e:
            import traceback
            logger.error(f"Agent run CRASHED: {e}")
            logger.error(traceback.format_exc())
            with contextlib.suppress(Exception):
                await send_event(RunErrorEvent(
                    type="RUN_ERROR",
                    message=f"Agent error: {str(e)}",
                    code="AGENT_CRASH",
                ))

    async def cancel_current() -> None:
        nonlocal agent_task
        if agent_task and not agent_task.done():
            steering.cancelled = True
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
                await cancel_current()
                steering = Steering()
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

            elif kind == "feedback":
                # Attach a user rating to the current run's Langfuse trace.
                if steering.trace_id and obs.is_enabled():
                    obs.score_trace(
                        trace_id=steering.trace_id,
                        name="user-feedback",
                        value=int(msg.get("score", 0)),
                        comment=msg.get("comment"),
                    )
                    obs.flush()
                await send_event(CustomEvent(type="CUSTOM", name="control.ack",
                                             value={"state": "feedback-recorded"}))

    except WebSocketDisconnect:
        logger.info("Client disconnected")
        await cancel_current()
        obs.flush()
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        obs.flush()