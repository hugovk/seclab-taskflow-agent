# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""``CopilotSDKBackend`` - the AgentBackend implementation that drives
the official ``copilot`` Python SDK.

Lifecycle
---------
* :meth:`build` starts a :class:`CopilotClient` (one per agent), creates
  a session bound to the agent's model and MCP servers, and wires a
  background queue that captures every ``SessionEvent``.
* :meth:`run_streamed` calls ``session.send`` and drains the queue
  yielding neutral :class:`TextDelta`/:class:`ToolStart`/:class:`ToolEnd`
  events until the session emits ``session.idle`` (success) or
  ``session.error`` (translated to :class:`BackendUnexpectedError`).
* :meth:`aclose` disconnects the session and stops the client.

Handoffs are not supported by the Copilot SDK - the capability matrix
already rejects multi-personality taskflows for this backend, so
:meth:`build` simply ignores ``spec.handoffs``.
"""

from __future__ import annotations

__all__ = ["CopilotSDKBackend"]

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ..base import AgentSpec, BackendCapabilities, StreamEvent, TextDelta, ToolEnd, ToolStart
from ..errors import BackendBadRequestError, BackendUnexpectedError
from .mcp import materialise_mcp_servers
from .permissions import build_permission_handler

_CAPABILITIES = BackendCapabilities(
    name="copilot_sdk",
    streaming=True,
    handoffs=False,
    custom_tools=True,
    mcp_stdio=True,
    mcp_sse=True,
    mcp_streamable_http=True,
    tool_use_behavior_exclude=False,
    parallel_tool_calls=False,
    temperature=False,
    reasoning_effort=True,
    responses_api=False,
)


@dataclass
class _CopilotHandle:
    """Opaque handle returned by :meth:`build`."""

    client: Any
    session: Any
    event_queue: asyncio.Queue[Any]


def _resolve_token(token_env: str | None) -> str | None:
    """Resolve the ``token_env`` field to an actual token string.

    The runner forwards the YAML token field unchanged, which can be
    either an env-var name or already the token itself. We try the env
    lookup first; if it fails we treat the value as the literal token,
    matching the openai-agents path's tolerance.
    """
    if not token_env:
        return None
    value = os.getenv(token_env)
    return value or token_env


def _reasoning_effort(model_settings: dict[str, Any]) -> str | None:
    raw = model_settings.get("reasoning_effort")
    if raw is None:
        return None
    if raw not in ("low", "medium", "high", "xhigh"):
        raise BackendBadRequestError(
            f"copilot_sdk: invalid reasoning_effort {raw!r} (expected low/medium/high/xhigh)"
        )
    return raw


class CopilotSDKBackend:
    """Adapter that drives the GitHub Copilot SDK."""

    capabilities: BackendCapabilities = _CAPABILITIES

    async def build(
        self,
        spec: AgentSpec,
        *,
        run_hooks: Any = None,  # noqa: ARG002 - openai-agents-shaped hooks have no parity yet
        agent_hooks: Any = None,  # noqa: ARG002
    ) -> _CopilotHandle:
        from copilot import CopilotClient, SubprocessConfig
        from copilot.session import SystemMessageReplaceConfig

        token = _resolve_token(spec.token_env)
        config = SubprocessConfig(github_token=token) if token else SubprocessConfig()
        client = CopilotClient(config, auto_start=False)
        await client.start()

        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue[Any] = asyncio.Queue()

        def _on_event(event: Any) -> None:
            # Called from the SDK's reader task; bounce safely onto our loop.
            with contextlib.suppress(RuntimeError):
                # loop may have closed during shutdown
                loop.call_soon_threadsafe(event_queue.put_nowait, event)

        try:
            session = await client.create_session(
                model=spec.model,
                on_permission_request=build_permission_handler(
                    spec.blocked_tools, headless=spec.headless
                ),
                streaming=True,
                mcp_servers=materialise_mcp_servers(spec.mcp_servers) or None,
                excluded_tools=spec.blocked_tools or None,
                reasoning_effort=_reasoning_effort(spec.model_settings),
                system_message=SystemMessageReplaceConfig(content=spec.instructions)
                if spec.instructions
                else None,
                on_event=_on_event,
            )
        except Exception:
            await client.stop()
            raise

        return _CopilotHandle(client=client, session=session, event_queue=event_queue)

    async def run_streamed(
        self,
        agent: _CopilotHandle,
        prompt: str,
        *,
        max_turns: int,  # noqa: ARG002 - the SDK manages turn limits internally
    ) -> AsyncIterator[StreamEvent]:
        from copilot.generated.session_events import SessionEventType

        await agent.session.send(prompt)
        agent_name = getattr(agent.session, "name", "copilot")

        while True:
            event = await agent.event_queue.get()
            etype = getattr(event, "type", None)
            data = getattr(event, "data", None)

            if etype in (
                SessionEventType.ASSISTANT_STREAMING_DELTA,
                SessionEventType.ASSISTANT_MESSAGE_DELTA,
            ):
                text = getattr(data, "content", None) or getattr(data, "delta", None) or ""
                if text:
                    yield TextDelta(text=text)
            elif etype == SessionEventType.TOOL_EXECUTION_START:
                yield ToolStart(
                    tool_name=getattr(data, "tool_name", "") or "",
                    agent_name=agent_name,
                )
            elif etype == SessionEventType.TOOL_EXECUTION_COMPLETE:
                yield ToolEnd(
                    tool_name=getattr(data, "tool_name", "") or "",
                    agent_name=agent_name,
                    result=str(getattr(data, "result", "") or ""),
                )
            elif etype == SessionEventType.SESSION_ERROR:
                msg = getattr(data, "message", None) or "copilot_sdk session error"
                raise BackendUnexpectedError(msg)
            elif etype == SessionEventType.SESSION_IDLE:
                return

    async def aclose(self, agent: _CopilotHandle) -> None:
        try:
            await agent.session.disconnect()
        except Exception:
            logging.exception("copilot_sdk: session.disconnect failed")
        try:
            await agent.client.stop()
        except Exception:
            logging.exception("copilot_sdk: client.stop failed")
