# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the capability gating helper."""

from __future__ import annotations

import pytest

from seclab_taskflow_agent.sdk.base import BackendCapabilities
from seclab_taskflow_agent.sdk.capabilities import assert_supported
from seclab_taskflow_agent.sdk.errors import BackendCapabilityError

_OPENAI = BackendCapabilities(
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

_COPILOT = BackendCapabilities(
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


def _call(caps: BackendCapabilities, **overrides) -> None:
    kwargs = {
        "agents_count": 1,
        "api_type": "chat_completions",
        "model_settings": {},
        "exclude_from_context": False,
    }
    kwargs.update(overrides)
    assert_supported(caps, **kwargs)


def test_openai_accepts_full_feature_matrix():
    _call(
        _OPENAI,
        agents_count=3,
        api_type="responses",
        model_settings={"temperature": 0.7},
        exclude_from_context=True,
    )


def test_copilot_accepts_minimal_task():
    _call(_COPILOT)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"agents_count": 2}, "agent handoffs"),
        ({"api_type": "responses"}, "api_type='responses'"),
        ({"model_settings": {"temperature": 0.1}}, "model_settings.temperature"),
        ({"exclude_from_context": True}, "exclude_from_context=true"),
    ],
)
def test_copilot_rejects_openai_only_features(overrides, field):
    with pytest.raises(BackendCapabilityError, match=field):
        _call(_COPILOT, **overrides)


def test_openai_rejects_reasoning_effort():
    with pytest.raises(BackendCapabilityError, match="reasoning_effort"):
        _call(_OPENAI, model_settings={"reasoning_effort": "high"})


def test_copilot_accepts_reasoning_effort():
    _call(_COPILOT, model_settings={"reasoning_effort": "medium"})


def test_error_message_names_backend():
    with pytest.raises(BackendCapabilityError, match="copilot_sdk"):
        _call(_COPILOT, agents_count=2)
