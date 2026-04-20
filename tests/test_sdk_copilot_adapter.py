# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the Copilot SDK adapter glue."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("copilot")

from copilot.generated.session_events import SessionEventType

from seclab_taskflow_agent.sdk import get_backend
from seclab_taskflow_agent.sdk.base import AgentSpec, MCPServerSpec, TextDelta
from seclab_taskflow_agent.sdk.copilot_sdk.backend import (
    CopilotSDKBackend,
    _reasoning_effort,
)
from seclab_taskflow_agent.sdk.errors import (
    BackendBadRequestError,
    BackendCapabilityError,
    BackendUnexpectedError,
)


def test_get_backend_returns_copilot_sdk_instance():
    backend = get_backend("copilot_sdk")
    assert isinstance(backend, CopilotSDKBackend)
    assert backend.name == "copilot_sdk"


def _spec(**overrides) -> AgentSpec:
    base = {
        "name": "a",
        "instructions": "",
        "model": "gpt-5-mini",
    }
    base.update(overrides)
    return AgentSpec(**base)


def test_validate_accepts_minimal_spec():
    CopilotSDKBackend().validate(_spec())


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"in_handoff_graph": True}, "handoffs"),
        ({"exclude_from_context": True}, "exclude_from_context"),
        ({"model_settings": {"temperature": 0.0}}, "temperature"),
        ({"model_settings": {"parallel_tool_calls": True}}, "parallel_tool_calls"),
    ],
)
def test_validate_rejects_unsupported(kwargs, field):
    backend = CopilotSDKBackend()
    with pytest.raises(BackendCapabilityError, match=field):
        backend.validate(_spec(**kwargs))


def test_validate_rejects_handoff_targets():
    backend = CopilotSDKBackend()
    with pytest.raises(BackendCapabilityError, match="handoffs"):
        backend.validate(_spec(handoffs=[_spec(name="b")]))


@dataclass
class _Event:
    type: SessionEventType
    data: Any = None


class _FakeSession:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, prompt: str) -> str:
        self.sent.append(prompt)
        return "req-1"


def test_run_streamed_translates_deltas_until_idle():
    async def _run():
        session = _FakeSession()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        for ev in (
            _Event(SessionEventType.ASSISTANT_MESSAGE_DELTA, SimpleNamespace(delta_content="hi ")),
            _Event(SessionEventType.ASSISTANT_MESSAGE_DELTA, SimpleNamespace(delta_content="there")),
            _Event(SessionEventType.SESSION_IDLE),
        ):
            queue.put_nowait(ev)
        handle = SimpleNamespace(client=None, session=session, event_queue=queue)
        backend = CopilotSDKBackend()
        return session, [
            event async for event in backend.run_streamed(handle, "go", max_turns=10)
        ]

    session, out = asyncio.run(_run())
    assert session.sent == ["go"]
    assert out == [TextDelta(text="hi "), TextDelta(text="there")]


def test_run_streamed_session_error_raises():
    async def _run():
        queue: asyncio.Queue[Any] = asyncio.Queue()
        queue.put_nowait(_Event(SessionEventType.SESSION_ERROR, SimpleNamespace(message="boom")))
        handle = SimpleNamespace(client=None, session=_FakeSession(), event_queue=queue)
        async for _ in CopilotSDKBackend().run_streamed(handle, "go", max_turns=10):
            pass

    with pytest.raises(BackendUnexpectedError, match="boom"):
        asyncio.run(_run())


def test_invalid_reasoning_effort_rejected():
    with pytest.raises(BackendBadRequestError, match="reasoning_effort"):
        _reasoning_effort({"reasoning_effort": "ludicrous"})


def test_aclose_handles_none():
    asyncio.run(CopilotSDKBackend().aclose(None))


def test_aclose_swallows_disconnect_failures():
    class _BadSession:
        async def disconnect(self) -> None:
            raise RuntimeError("nope")

    class _Client:
        async def stop(self) -> None:
            return None

    handle = SimpleNamespace(client=_Client(), session=_BadSession(), event_queue=asyncio.Queue())
    asyncio.run(CopilotSDKBackend().aclose(handle))


def test_mcp_specs_are_used():
    # Sanity: MCPServerSpec is what build() consumes via build_mcp_config.
    spec = _spec(mcp_servers=[MCPServerSpec(name="m", kind="stdio", params={"command": "x"})])
    CopilotSDKBackend().validate(spec)
