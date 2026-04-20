# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the backend-neutral SDK abstraction skeleton."""

from __future__ import annotations

import pytest

from seclab_taskflow_agent import sdk
from seclab_taskflow_agent.sdk.base import (
    AgentSpec,
    BackendCapabilities,
    MCPServerSpec,
    StreamEvent,
    TextDelta,
    ToolEnd,
    ToolSpec,
    ToolStart,
)

_OPENAI_CAPS = BackendCapabilities(
    name="openai_agents",
    streaming=True,
    handoffs=True,
    custom_tools=False,
    mcp_stdio=True,
    mcp_sse=True,
    mcp_streamable_http=True,
    tool_use_behavior_exclude=True,
    parallel_tool_calls=True,
    temperature=True,
    reasoning_effort=False,
    responses_api=True,
)

_COPILOT_CAPS = BackendCapabilities(
    name="copilot_sdk",
    streaming=True,
    handoffs=False,
    custom_tools=True,
    mcp_stdio=True,
    mcp_sse=False,
    mcp_streamable_http=False,
    tool_use_behavior_exclude=False,
    parallel_tool_calls=False,
    temperature=False,
    reasoning_effort=True,
    responses_api=False,
)


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    monkeypatch.setattr(sdk, "_CAPABILITIES", {})
    monkeypatch.setattr(sdk, "_BACKENDS", {})
    monkeypatch.delenv("SECLAB_TASKFLOW_BACKEND", raising=False)


def test_register_and_lookup_capabilities():
    sdk.register_backend_capabilities(_OPENAI_CAPS)
    assert sdk.available_backends() == ("openai_agents",)
    assert sdk.get_backend_capabilities("openai_agents") is _OPENAI_CAPS


def test_register_rejects_unknown_backend_name():
    bad = BackendCapabilities(
        name="not_a_backend",  # type: ignore[arg-type]
        streaming=False, handoffs=False, custom_tools=False,
        mcp_stdio=False, mcp_sse=False, mcp_streamable_http=False,
        tool_use_behavior_exclude=False, parallel_tool_calls=False,
        temperature=False, reasoning_effort=False, responses_api=False,
    )
    with pytest.raises(ValueError, match="Unknown backend name"):
        sdk.register_backend_capabilities(bad)


def test_get_capabilities_unknown_raises():
    with pytest.raises(ValueError, match="not registered"):
        sdk.get_backend_capabilities("openai_agents")


def test_resolve_explicit_wins(monkeypatch):
    sdk.register_backend_capabilities(_OPENAI_CAPS)
    sdk.register_backend_capabilities(_COPILOT_CAPS)
    monkeypatch.setenv("SECLAB_TASKFLOW_BACKEND", "copilot_sdk")
    assert sdk.resolve_backend_name(explicit="openai_agents") == "openai_agents"


def test_resolve_env_beats_auto_default(monkeypatch):
    sdk.register_backend_capabilities(_OPENAI_CAPS)
    sdk.register_backend_capabilities(_COPILOT_CAPS)
    monkeypatch.setenv("SECLAB_TASKFLOW_BACKEND", "openai_agents")
    assert sdk.resolve_backend_name(endpoint="https://api.githubcopilot.com") == "openai_agents"


def test_resolve_auto_default_copilot_endpoint():
    sdk.register_backend_capabilities(_OPENAI_CAPS)
    sdk.register_backend_capabilities(_COPILOT_CAPS)
    assert sdk.resolve_backend_name(endpoint="https://api.githubcopilot.com/") == "copilot_sdk"


def test_resolve_auto_default_without_copilot_registered():
    sdk.register_backend_capabilities(_OPENAI_CAPS)
    # Copilot endpoint but SDK adapter never registered → stay on openai_agents.
    assert sdk.resolve_backend_name(endpoint="https://api.githubcopilot.com") == "openai_agents"


def test_resolve_default_is_openai_agents():
    sdk.register_backend_capabilities(_OPENAI_CAPS)
    assert sdk.resolve_backend_name() == "openai_agents"


def test_resolve_unknown_candidate_raises():
    sdk.register_backend_capabilities(_OPENAI_CAPS)
    with pytest.raises(ValueError, match="not available"):
        sdk.resolve_backend_name(explicit="copilot_sdk")


def test_stream_event_union_covers_all_variants():
    events: list[StreamEvent] = [
        TextDelta("hi"),
        ToolStart(tool_name="t", agent_name="a"),
        ToolEnd(tool_name="t", agent_name="a", result="ok"),
    ]
    assert [type(e).__name__ for e in events] == ["TextDelta", "ToolStart", "ToolEnd"]


def test_agent_spec_defaults():
    spec = AgentSpec(name="n", instructions="", model="gpt-5")
    assert spec.handoffs == []
    assert spec.tools == []
    assert spec.mcp_servers == []
    assert spec.api_type is None
    assert spec.exclude_from_context is False


def test_mcp_and_tool_spec_are_immutable():
    from dataclasses import FrozenInstanceError

    mcp = MCPServerSpec(name="tb", kind="stdio", params={"command": "x"})
    tool = ToolSpec(name="t", description="", parameters={}, handler=lambda _x: "")
    with pytest.raises(FrozenInstanceError):
        mcp.name = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        tool.name = "other"  # type: ignore[misc]


class _FakeBackend:
    """Minimal backend for testing registry wiring."""

    def __init__(self, caps):
        self.capabilities = caps

    async def build(self, spec, *, run_hooks=None, agent_hooks=None):  # noqa: ARG002  # pragma: no cover
        return object()

    def run_streamed(self, agent, prompt, *, max_turns):  # noqa: ARG002  # pragma: no cover
        raise NotImplementedError

    async def aclose(self, agent):  # noqa: ARG002  # pragma: no cover
        return None


def test_register_backend_also_registers_capabilities():
    fake = _FakeBackend(_OPENAI_CAPS)
    sdk.register_backend(fake)
    assert sdk.get_backend("openai_agents") is fake
    assert sdk.get_backend_capabilities("openai_agents") is _OPENAI_CAPS
    assert sdk.available_backends() == ("openai_agents",)


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError, match="not registered"):
        sdk.get_backend("openai_agents")


def test_fake_backend_satisfies_protocol():
    from seclab_taskflow_agent.sdk.base import AgentBackend

    assert isinstance(_FakeBackend(_OPENAI_CAPS), AgentBackend)
