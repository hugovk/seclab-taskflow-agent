# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Cross-backend smoke tests.

These tests parametrise over every adapter registered via the
``sdk.get_backend`` registry to ensure each one exposes a coherent
capability descriptor and a working ``aclose`` no-op on a ``None``
handle. They are deliberately lightweight: full integration coverage
lives in ``test_sdk_openai_adapter.py`` and ``test_sdk_copilot_adapter.py``.
"""

from __future__ import annotations

import asyncio

import pytest

from seclab_taskflow_agent import sdk
from seclab_taskflow_agent.sdk.capabilities import assert_supported
from seclab_taskflow_agent.sdk.errors import BackendCapabilityError


def _all_backends() -> list[str]:
    # Force-load the optional adapter so it is registered for the test
    # session if its dependency is installed.
    for name in ("openai_agents", "copilot_sdk"):
        sdk._autoload_backend(name)
    return sorted(sdk.available_backends())


BACKENDS = _all_backends()


@pytest.mark.parametrize("name", BACKENDS)
def test_backend_capabilities_have_consistent_name(name):
    caps = sdk.get_backend_capabilities(name)
    assert caps.name == name


@pytest.mark.parametrize("name", BACKENDS)
def test_backend_instance_capabilities_match_registry(name):
    backend = sdk.get_backend(name)
    assert backend.capabilities == sdk.get_backend_capabilities(name)


@pytest.mark.parametrize("name", BACKENDS)
def test_backend_accepts_minimal_task(name):
    caps = sdk.get_backend_capabilities(name)
    assert_supported(
        caps,
        agents_count=1,
        api_type="chat_completions",
        model_settings={},
        exclude_from_context=False,
    )


@pytest.mark.parametrize("name", BACKENDS)
def test_handoff_gating_matches_capability(name):
    caps = sdk.get_backend_capabilities(name)
    if caps.handoffs:
        assert_supported(
            caps,
            agents_count=2,
            api_type="chat_completions",
            model_settings={},
            exclude_from_context=False,
        )
    else:
        with pytest.raises(BackendCapabilityError, match="handoffs"):
            assert_supported(
                caps,
                agents_count=2,
                api_type="chat_completions",
                model_settings={},
                exclude_from_context=False,
            )


def test_resolver_round_trip_for_each_backend():
    for name in BACKENDS:
        assert sdk.resolve_backend_name(explicit=name) == name


@pytest.mark.parametrize("name", BACKENDS)
def test_aclose_handles_no_op_handle(name):
    backend = sdk.get_backend(name)

    async def _run():
        await backend.aclose(None)

    asyncio.run(_run())
