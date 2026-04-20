# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the backend-neutral hook classes."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from seclab_taskflow_agent.sdk.hooks import HookContext, TaskAgentHooks, TaskRunHooks


def test_run_hooks_fire_all_callbacks():
    events: list[tuple] = []

    async def on_agent_start(ctx):
        events.append(("agent_start", ctx.agent_name))

    async def on_agent_end(ctx, output):
        events.append(("agent_end", ctx.agent_name, output))

    async def on_tool_start(ctx, tool_name):
        events.append(("tool_start", ctx.agent_name, tool_name))

    async def on_tool_end(ctx, tool_name, result):
        events.append(("tool_end", ctx.agent_name, tool_name, result))

    hooks = TaskRunHooks(
        on_agent_start=on_agent_start,
        on_agent_end=on_agent_end,
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
    )

    async def drive():
        ctx = HookContext(agent_name="A")
        await hooks.on_agent_start(ctx)
        await hooks.on_agent_end(ctx, "out")
        await hooks.on_tool_start(ctx, "tool1")
        await hooks.on_tool_end(ctx, "tool1", "ok")

    asyncio.run(drive())

    assert events == [
        ("agent_start", "A"),
        ("agent_end", "A", "out"),
        ("tool_start", "A", "tool1"),
        ("tool_end", "A", "tool1", "ok"),
    ]


def test_run_hooks_without_callbacks_are_no_ops():
    hooks = TaskRunHooks()

    async def drive():
        ctx = HookContext(agent_name="A")
        await hooks.on_agent_start(ctx)
        await hooks.on_agent_end(ctx, None)
        await hooks.on_tool_start(ctx, "t")
        await hooks.on_tool_end(ctx, "t", "r")

    asyncio.run(drive())  # must not raise


def test_agent_hooks_on_handoff_passes_source():
    captured: list[tuple[str, str]] = []

    async def on_handoff(ctx, source_name):
        captured.append((ctx.agent_name, source_name))

    hooks = TaskAgentHooks(on_handoff=on_handoff)
    asyncio.run(hooks.on_handoff(HookContext(agent_name="target"), "source"))
    assert captured == [("target", "source")]


def test_agent_hooks_without_callbacks_are_no_ops():
    hooks = TaskAgentHooks()

    async def drive():
        ctx = HookContext(agent_name="A")
        await hooks.on_start(ctx)
        await hooks.on_end(ctx, "done")
        await hooks.on_tool_start(ctx, "t")
        await hooks.on_tool_end(ctx, "t", "r")
        await hooks.on_handoff(ctx, "source")

    asyncio.run(drive())


def test_hook_context_is_frozen():
    ctx = HookContext(agent_name="A", raw={"some": "obj"})
    with pytest.raises(FrozenInstanceError):
        ctx.agent_name = "B"  # type: ignore[misc]
    assert ctx.raw == {"some": "obj"}
