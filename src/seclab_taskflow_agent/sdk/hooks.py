# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Backend-neutral hook protocol for agent lifecycle callbacks.

Adapters translate their SDK's native hook signatures into these
classes so the runner can observe tool starts/ends, agent handoffs,
and final outputs without importing any backend SDK.
"""

from __future__ import annotations

__all__ = [
    "HookContext",
    "TaskAgentHooks",
    "TaskRunHooks",
]

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HookContext:
    """Opaque context passed to every hook callback.

    ``agent_name`` is always populated. ``raw`` carries the backend-native
    context object (e.g. ``RunContextWrapper`` for openai-agents, or the
    session object for the Copilot SDK) for advanced callers that need
    backend-specific behaviour.
    """

    agent_name: str
    raw: Any = None


_Callback = Callable[..., Any]


class TaskRunHooks:
    """Observes the whole runner lifetime, including agent handoffs.

    All callbacks are optional and coroutines. Missing callbacks are
    silently skipped so that adapters can wire them up unconditionally.
    """

    def __init__(
        self,
        on_agent_start: _Callback | None = None,
        on_agent_end: _Callback | None = None,
        on_tool_start: _Callback | None = None,
        on_tool_end: _Callback | None = None,
    ) -> None:
        self._on_agent_start = on_agent_start
        self._on_agent_end = on_agent_end
        self._on_tool_start = on_tool_start
        self._on_tool_end = on_tool_end

    async def on_agent_start(self, ctx: HookContext) -> None:
        logging.debug(f"TaskRunHooks on_agent_start: {ctx.agent_name}")
        if self._on_agent_start:
            await self._on_agent_start(ctx)

    async def on_agent_end(self, ctx: HookContext, output: Any) -> None:
        logging.debug(f"TaskRunHooks on_agent_end: {ctx.agent_name}")
        if self._on_agent_end:
            await self._on_agent_end(ctx, output)

    async def on_tool_start(self, ctx: HookContext, tool_name: str) -> None:
        logging.debug(f"TaskRunHooks on_tool_start: {tool_name}")
        if self._on_tool_start:
            await self._on_tool_start(ctx, tool_name)

    async def on_tool_end(self, ctx: HookContext, tool_name: str, result: str) -> None:
        logging.debug(f"TaskRunHooks on_tool_end: {tool_name}")
        if self._on_tool_end:
            await self._on_tool_end(ctx, tool_name, result)


class TaskAgentHooks:
    """Observes the lifetime of a single agent (no handoffs)."""

    def __init__(
        self,
        on_handoff: _Callback | None = None,
        on_start: _Callback | None = None,
        on_end: _Callback | None = None,
        on_tool_start: _Callback | None = None,
        on_tool_end: _Callback | None = None,
    ) -> None:
        self._on_handoff = on_handoff
        self._on_start = on_start
        self._on_end = on_end
        self._on_tool_start = on_tool_start
        self._on_tool_end = on_tool_end

    async def on_handoff(self, ctx: HookContext, source_name: str) -> None:
        logging.debug(f"TaskAgentHooks on_handoff: {source_name} -> {ctx.agent_name}")
        if self._on_handoff:
            await self._on_handoff(ctx, source_name)

    async def on_start(self, ctx: HookContext) -> None:
        logging.debug(f"TaskAgentHooks on_start: {ctx.agent_name}")
        if self._on_start:
            await self._on_start(ctx)

    async def on_end(self, ctx: HookContext, output: Any) -> None:
        logging.debug(f"TaskAgentHooks on_end: {ctx.agent_name}")
        if self._on_end:
            await self._on_end(ctx, output)

    async def on_tool_start(self, ctx: HookContext, tool_name: str) -> None:
        logging.debug(f"TaskAgentHooks on_tool_start: {tool_name}")
        if self._on_tool_start:
            await self._on_tool_start(ctx, tool_name)

    async def on_tool_end(self, ctx: HookContext, tool_name: str, result: str) -> None:
        logging.debug(f"TaskAgentHooks on_tool_end: {tool_name}")
        if self._on_tool_end:
            await self._on_tool_end(ctx, tool_name, result)
