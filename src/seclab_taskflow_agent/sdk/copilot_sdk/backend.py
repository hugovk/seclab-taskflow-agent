# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""GitHub Copilot SDK backend adapter.

* :meth:`build` starts a :class:`CopilotClient`, creates a session bound
  to the agent's model and MCP servers, and wires a queue that captures
  every ``SessionEvent``.
* :meth:`run_streamed` calls ``session.send`` and drains the queue
  yielding :class:`TextDelta` until ``session.idle`` (success) or
  ``session.error`` (translated to :class:`BackendUnexpectedError`).
* :meth:`aclose` disconnects the session and stops the client.
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

from ..base import AgentSpec, StreamEvent, TextDelta, ToolEnd
from ..errors import BackendBadRequestError, BackendCapabilityError, BackendUnexpectedError
from .mcp import build_mcp_config
from .permissions import build_permission_handler

_VALID_REASONING = ("low", "medium", "high", "xhigh")


@dataclass
class _CopilotHandle:
    client: Any
    session: Any
    event_queue: asyncio.Queue[Any]


def _resolve_token(token_env: str | None) -> str | None:
    if not token_env:
        return None
    return os.getenv(token_env) or token_env


def _normalize_model(model: str) -> str:
    """Strip provider prefix (e.g. ``openai/gpt-4.1`` → ``gpt-4.1``).

    The Copilot SDK selects models by bare name; the provider-prefixed form
    is a CAPI inference convention used by the openai-agents path.
    """
    return model.split("/", 1)[1] if "/" in model else model


def _tool_result_text(data: Any) -> str | None:
    """Extract the textual payload from a ``TOOL_EXECUTION_COMPLETE`` event.

    The Copilot SDK delivers MCP tool results as a ``Result`` object with a
    list of ``ContentElement`` entries. We concatenate the ``text`` fields
    of the text elements, mirroring how openai-agents stringifies an MCP
    text content list before handing it to the runner's tool-end hook.
    """
    result = getattr(data, "result", None)
    if result is None:
        return None
    contents = getattr(result, "contents", None) or []
    parts = [getattr(c, "text", "") for c in contents if getattr(c, "text", None)]
    if parts:
        return "".join(parts)
    content = getattr(result, "content", None)
    return content if isinstance(content, str) else None


def _reasoning_effort(model_settings: dict[str, Any]) -> str | None:
    raw = model_settings.get("reasoning_effort")
    if raw is None:
        return None
    if raw not in _VALID_REASONING:
        raise BackendBadRequestError(
            f"copilot_sdk: invalid reasoning_effort {raw!r} "
            f"(expected one of {_VALID_REASONING})"
        )
    return raw


class CopilotSDKBackend:
    """Adapter that drives the GitHub Copilot SDK."""

    name = "copilot_sdk"

    def validate(self, spec: AgentSpec) -> None:
        """Reject YAML fields the Copilot SDK cannot honour.

        The SDK has no concept of agent-to-agent handoffs, no
        ``parallel_tool_calls`` or ``temperature`` knob on the
        underlying client, and no equivalent of openai-agents'
        ``exclude_from_context`` tool-output suppression. ``api_type``
        is silently ignored — Copilot picks its own wire protocol per
        model.
        """
        if spec.handoffs or spec.in_handoff_graph:
            raise BackendCapabilityError(
                "copilot_sdk: agent handoffs are not supported"
            )
        if spec.exclude_from_context:
            raise BackendCapabilityError(
                "copilot_sdk: exclude_from_context is not supported"
            )
        for unsupported in ("temperature", "parallel_tool_calls"):
            if unsupported in spec.model_settings:
                raise BackendCapabilityError(
                    f"copilot_sdk: model_settings.{unsupported} is not supported"
                )

    async def build(
        self,
        spec: AgentSpec,
        *,
        run_hooks: Any = None,
        agent_hooks: Any = None,
    ) -> _CopilotHandle:
        del run_hooks, agent_hooks  # no parity yet
        from copilot import CopilotClient, SubprocessConfig
        from copilot.session import SystemMessageReplaceConfig

        token = _resolve_token(spec.token_env)
        config = SubprocessConfig(github_token=token) if token else SubprocessConfig()
        client = CopilotClient(config, auto_start=False)
        await client.start()

        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue[Any] = asyncio.Queue()

        def _on_event(event: Any) -> None:
            with contextlib.suppress(RuntimeError):
                # Loop may have closed during shutdown.
                loop.call_soon_threadsafe(event_queue.put_nowait, event)

        try:
            session = await client.create_session(
                model=_normalize_model(spec.model),
                on_permission_request=build_permission_handler(
                    spec.blocked_tools, headless=spec.headless
                ),
                streaming=True,
                mcp_servers=build_mcp_config(spec.mcp_servers) or None,
                excluded_tools=spec.blocked_tools or None,
                reasoning_effort=_reasoning_effort(spec.model_settings),
                system_message=(
                    SystemMessageReplaceConfig(content=spec.instructions)
                    if spec.instructions
                    else None
                ),
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
        max_turns: int,
    ) -> AsyncIterator[StreamEvent]:
        del max_turns  # the SDK manages turn limits internally
        from copilot.generated.session_events import SessionEventType

        await agent.session.send(prompt)

        while True:
            event = await agent.event_queue.get()
            etype = getattr(event, "type", None)
            data = getattr(event, "data", None)

            if etype == SessionEventType.ASSISTANT_MESSAGE_DELTA:
                text = getattr(data, "delta_content", None) or ""
                if text:
                    yield TextDelta(text=text)
            elif etype == SessionEventType.TOOL_EXECUTION_COMPLETE:
                if getattr(data, "success", False):
                    text = _tool_result_text(data)
                    if text is not None:
                        yield ToolEnd(
                            tool_name=getattr(data, "tool_name", "") or "",
                            text=text,
                        )
            elif etype == SessionEventType.SESSION_ERROR:
                raise BackendUnexpectedError(
                    getattr(data, "message", None) or "copilot_sdk session error"
                )
            elif etype == SessionEventType.SESSION_IDLE:
                return

    async def aclose(self, agent: _CopilotHandle | None) -> None:
        if agent is None:
            return
        try:
            await agent.session.disconnect()
        except Exception:
            logging.exception("copilot_sdk: session.disconnect failed")
        try:
            await agent.client.stop()
        except Exception:
            logging.exception("copilot_sdk: client.stop failed")
