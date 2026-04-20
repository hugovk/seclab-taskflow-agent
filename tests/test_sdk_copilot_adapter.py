# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``sdk.copilot_sdk.backend``.

The Copilot SDK's transport is mocked: we only verify the adapter's
glue logic (capability descriptor, build → run_streamed → aclose
sequence, event translation, exception mapping).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("copilot")

from copilot.generated.session_events import SessionEventType

from seclab_taskflow_agent.sdk import get_backend, get_backend_capabilities
from seclab_taskflow_agent.sdk.base import (
    TextDelta,
    ToolEnd,
    ToolStart,
)
from seclab_taskflow_agent.sdk.copilot_sdk.backend import CopilotSDKBackend
from seclab_taskflow_agent.sdk.errors import BackendBadRequestError, BackendUnexpectedError


def test_capability_descriptor_registered():
    caps = get_backend_capabilities("copilot_sdk")
    assert caps.name == "copilot_sdk"
    assert caps.streaming is True
    assert caps.handoffs is False
    assert caps.reasoning_effort is True
    assert caps.temperature is False


def test_get_backend_returns_copilot_sdk_instance():
    backend = get_backend("copilot_sdk")
    assert isinstance(backend, CopilotSDKBackend)


@dataclass
class _Event:
    type: SessionEventType
    data: Any = None


class _FakeSession:
    def __init__(self, events: list[_Event]) -> None:
        self.name = "agent"
        self._events = events
        self.sent: list[str] = []
        self.disconnected = False

    async def send(self, prompt: str) -> str:
        self.sent.append(prompt)
        return "req-1"

    async def disconnect(self) -> None:
        self.disconnected = True


async def _drain(backend, handle, prompt):
    return [event async for event in backend.run_streamed(handle, prompt, max_turns=10)]


def test_run_streamed_translates_events_until_idle():
    async def _run():
        events = [
            _Event(SessionEventType.ASSISTANT_STREAMING_DELTA, SimpleNamespace(content="hello ")),
            _Event(SessionEventType.ASSISTANT_STREAMING_DELTA, SimpleNamespace(content="world")),
            _Event(SessionEventType.TOOL_EXECUTION_START, SimpleNamespace(tool_name="search")),
            _Event(
                SessionEventType.TOOL_EXECUTION_COMPLETE,
                SimpleNamespace(tool_name="search", result="ok"),
            ),
            _Event(SessionEventType.SESSION_IDLE),
        ]
        session = _FakeSession(events)
        queue: asyncio.Queue[Any] = asyncio.Queue()
        for ev in events:
            queue.put_nowait(ev)
        handle = SimpleNamespace(client=None, session=session, event_queue=queue)
        backend = CopilotSDKBackend()
        out = await _drain(backend, handle, "go")
        return session, out

    session, out = asyncio.run(_run())
    assert session.sent == ["go"]
    assert out == [
        TextDelta(text="hello "),
        TextDelta(text="world"),
        ToolStart(tool_name="search", agent_name="agent"),
        ToolEnd(tool_name="search", agent_name="agent", result="ok"),
    ]


def test_run_streamed_session_error_raises():
    async def _run():
        events = [
            _Event(SessionEventType.SESSION_ERROR, SimpleNamespace(message="boom")),
        ]
        queue: asyncio.Queue[Any] = asyncio.Queue()
        for ev in events:
            queue.put_nowait(ev)
        handle = SimpleNamespace(
            client=None, session=_FakeSession(events), event_queue=queue
        )
        backend = CopilotSDKBackend()
        await _drain(backend, handle, "go")

    with pytest.raises(BackendUnexpectedError, match="boom"):
        asyncio.run(_run())


def test_invalid_reasoning_effort_rejected():
    from seclab_taskflow_agent.sdk.copilot_sdk.backend import _reasoning_effort

    with pytest.raises(BackendBadRequestError, match="reasoning_effort"):
        _reasoning_effort({"reasoning_effort": "ludicrous"})


def test_aclose_swallows_disconnect_failures():
    class _BadSession:
        async def disconnect(self):
            raise RuntimeError("nope")

    class _Client:
        async def stop(self):
            return None

    async def _run():
        handle = SimpleNamespace(
            client=_Client(), session=_BadSession(), event_queue=asyncio.Queue()
        )
        backend = CopilotSDKBackend()
        await backend.aclose(handle)

    asyncio.run(_run())
