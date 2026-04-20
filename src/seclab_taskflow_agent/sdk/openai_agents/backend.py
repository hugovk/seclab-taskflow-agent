# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""openai-agents backend adapter.

Wraps :class:`seclab_taskflow_agent.agent.TaskAgent` so the runner can
drive it through the backend-neutral SDK interface. The adapter is
responsible for three things:

1. Translating :class:`AgentSpec` into the constructor kwargs that
   ``TaskAgent`` expects.
2. Yielding :class:`StreamEvent` values from the ``stream_events()``
   iterator exposed by the openai-agents ``Runner``.
3. Mapping openai-agents / OpenAI client exceptions to the neutral
   :mod:`sdk.errors` hierarchy.
"""

from __future__ import annotations

__all__ = ["OpenAIAgentsBackend"]

from collections.abc import AsyncIterator
from typing import Any

from agents.exceptions import AgentsException, MaxTurnsExceeded
from openai import APIConnectionError, APITimeoutError, BadRequestError, RateLimitError
from openai.types.responses import ResponseTextDeltaEvent

from ..base import (
    AgentHandle,
    AgentSpec,
    BackendCapabilities,
    StreamEvent,
    TextDelta,
)
from ..errors import (
    BackendBadRequestError,
    BackendMaxTurnsError,
    BackendRateLimitError,
    BackendTimeoutError,
    BackendUnexpectedError,
)

_CAPABILITIES = BackendCapabilities(
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


class OpenAIAgentsBackend:
    """Adapter that drives the openai-agents SDK."""

    capabilities: BackendCapabilities = _CAPABILITIES

    async def build(
        self,
        spec: AgentSpec,
        *,
        run_hooks: Any = None,
        agent_hooks: Any = None,
    ) -> AgentHandle:
        """Construct a :class:`TaskAgent` from *spec*.

        ``spec.mcp_servers`` is currently treated as an opaque list of
        already-materialised ``agents.mcp.MCPServer`` instances passed
        through ``params["_native"]``. The runner-decoupling slice will
        add proper MCP materialisation on top of this.
        """
        # Imported lazily to avoid importing agent.py (and thus the
        # OpenAI client wiring) when only the registry surface is used.
        from agents.agent import ModelSettings

        from ...agent import TaskAgent

        mcp_servers = [s.params["_native"] for s in spec.mcp_servers if "_native" in s.params]
        handoffs = [
            (await self.build(h, run_hooks=run_hooks, agent_hooks=agent_hooks)).agent  # type: ignore[attr-defined]
            for h in spec.handoffs
        ]
        return TaskAgent(
            name=spec.name,
            instructions=spec.instructions,
            handoffs=handoffs,
            exclude_from_context=spec.exclude_from_context,
            mcp_servers=mcp_servers,
            model=spec.model,
            model_settings=ModelSettings(**spec.model_settings) if spec.model_settings else None,
            api_type=spec.api_type or "chat_completions",
            endpoint=spec.endpoint,
            token=spec.token_env,
            run_hooks=run_hooks,
            agent_hooks=agent_hooks,
        )

    async def run_streamed(
        self,
        agent: AgentHandle,
        prompt: str,
        *,
        max_turns: int,
    ) -> AsyncIterator[StreamEvent]:
        """Yield neutral stream events from the agent's streamed run.

        Translates openai-agents raw response events to :class:`TextDelta`.
        Other raw event types are ignored by design — the runner only
        needs text deltas to drive its terminal output.
        """
        try:
            result = agent.run_streamed(prompt, max_turns=max_turns)  # type: ignore[attr-defined]
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    yield TextDelta(text=event.data.delta)
        except MaxTurnsExceeded as exc:
            raise BackendMaxTurnsError(str(exc)) from exc
        except RateLimitError as exc:
            raise BackendRateLimitError(str(exc)) from exc
        except (APITimeoutError, APIConnectionError) as exc:
            raise BackendTimeoutError(str(exc)) from exc
        except BadRequestError as exc:
            raise BackendBadRequestError(str(exc)) from exc
        except AgentsException as exc:
            raise BackendUnexpectedError(str(exc)) from exc

    async def aclose(self, agent: AgentHandle) -> None:
        """No-op: openai-agents agents do not hold resources themselves.

        MCP server cleanup is owned by the runner (see
        ``mcp_lifecycle.mcp_session_task``) and is not the adapter's
        responsibility in this revision.
        """
