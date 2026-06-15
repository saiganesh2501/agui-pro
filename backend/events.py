"""
events.py — re-exports the OFFICIAL AG-UI event classes from ag_ui.core.

We do NOT redefine the protocol here. We import the genuine, Pydantic-based
event types from the installed `ag-ui-protocol` package so every event we emit
is spec-correct and serializes to the official camelCase wire format.

Install:  pip install ag-ui-protocol
Docs:     https://docs.ag-ui.com/sdk/python/core/events

If your installed SDK version names an event differently and an import below
fails, the ImportError will name it precisely — adjust that single name.
"""
from __future__ import annotations

# Core enum + base + the events we emit. These names are stable across the
# 0.1.x line of ag-ui-protocol.
from ag_ui.core import (  # type: ignore
    EventType,
    BaseEvent,
    # lifecycle
    RunStartedEvent,
    RunFinishedEvent,
    RunErrorEvent,
    StepStartedEvent,
    StepFinishedEvent,
    # text
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    # tool calls
    ToolCallStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    # state
    StateSnapshotEvent,
    StateDeltaEvent,
    # extension
    CustomEvent,
    RawEvent,
    # inbound payload
    RunAgentInput,
)

__all__ = [
    "EventType", "BaseEvent",
    "RunStartedEvent", "RunFinishedEvent", "RunErrorEvent",
    "StepStartedEvent", "StepFinishedEvent",
    "TextMessageStartEvent", "TextMessageContentEvent", "TextMessageEndEvent",
    "ToolCallStartEvent", "ToolCallArgsEvent", "ToolCallEndEvent", "ToolCallResultEvent",
    "StateSnapshotEvent", "StateDeltaEvent",
    "CustomEvent", "RawEvent", "RunAgentInput",
]
