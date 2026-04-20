# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Backend-neutral types for the agent SDK abstraction.

Defines the contract that every backend adapter must implement so the
runner can drive either ``openai-agents`` or the GitHub Copilot SDK
through the same code path. No backend-specific imports are allowed in
this module.
"""

from __future__ import annotations

__all__ = [
    "AgentBackend",
    "AgentHandle",
    "AgentSpec",
    "BackendCapabilities",
    "BackendName",
    "MCPServerSpec",
    "RunResult",
    "StreamEvent",
    "TextDelta",
    "ToolEnd",
    "ToolSpec",
    "ToolStart",
]

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

BackendName = Literal["openai_agents", "copilot_sdk"]


@dataclass(frozen=True)
class BackendCapabilities:
    """Static description of what a backend supports.

    The runner consults this descriptor at task materialisation to fail
    fast on YAML features the active backend cannot honour.
    """

    name: BackendName
    streaming: bool
    handoffs: bool
    custom_tools: bool
    mcp_stdio: bool
    mcp_sse: bool
    mcp_streamable_http: bool
    tool_use_behavior_exclude: bool
    parallel_tool_calls: bool
    temperature: bool
    reasoning_effort: bool
    responses_api: bool


@dataclass(frozen=True)
class MCPServerSpec:
    """Backend-neutral MCP server descriptor."""

    name: str
    kind: Literal["stdio", "sse", "streamable"]
    params: dict[str, Any]
    confirms: list[str] = field(default_factory=list)
    server_prompt: str = ""
    client_session_timeout: float = 0.0
    reconnecting: bool = False


@dataclass(frozen=True)
class ToolSpec:
    """Backend-neutral custom tool descriptor."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Any  # Callable[[dict[str, Any]], Awaitable[str] | str]
    overrides_built_in: bool = False


@dataclass
class AgentSpec:
    """Backend-neutral agent configuration.

    Mirrors what the runner needs to construct an agent: identity,
    instructions, model selection, tools/MCP, and an optional handoff
    graph. Values that have no meaning for a given backend (for example
    ``api_type`` on the Copilot SDK) are ignored by that adapter.
    """

    name: str
    instructions: str
    model: str
    model_settings: dict[str, Any] = field(default_factory=dict)
    tools: list[ToolSpec] = field(default_factory=list)
    mcp_servers: list[MCPServerSpec] = field(default_factory=list)
    handoffs: list[AgentSpec] = field(default_factory=list)
    exclude_from_context: bool = False
    api_type: str | None = None
    endpoint: str | None = None
    token_env: str | None = None
    # Set to True when this agent participates in a multi-personality
    # handoff graph. Adapters may use it to apply backend-specific
    # prompt scaffolding (openai-agents prepends handoff instructions to
    # every participant, including the handoff targets themselves).
    in_handoff_graph: bool = False
    # Tool names the user wants disabled for this agent. The
    # openai-agents backend pre-filters at MCP-server build time and
    # ignores this field; the Copilot SDK adapter forwards the list as
    # ``excluded_tools`` to ``create_session``.
    blocked_tools: list[str] = field(default_factory=list)
    # When True, interactive permission prompts must auto-approve
    # instead of blocking on user input. The Copilot SDK adapter wires
    # this into the ``on_permission_request`` callback.
    headless: bool = False


@dataclass(frozen=True)
class TextDelta:
    """Incremental text chunk emitted while the model is generating."""

    text: str


@dataclass(frozen=True)
class ToolStart:
    """Emitted before a tool invocation begins."""

    tool_name: str
    agent_name: str


@dataclass(frozen=True)
class ToolEnd:
    """Emitted after a tool invocation completes."""

    tool_name: str
    agent_name: str
    result: str


StreamEvent = TextDelta | ToolStart | ToolEnd


@dataclass
class RunResult:
    """Outcome of a non-streaming run."""

    final_output: str
    completed: bool


class AgentHandle(Protocol):
    """Opaque handle returned by :meth:`AgentBackend.build`.

    Backends are free to use their native agent/session object here.
    The runner only passes it back into ``run_streamed`` and ``aclose``.
    """


@runtime_checkable
class AgentBackend(Protocol):
    """The contract every backend adapter implements."""

    capabilities: BackendCapabilities

    async def build(
        self,
        spec: AgentSpec,
        *,
        run_hooks: Any = None,
        agent_hooks: Any = None,
    ) -> AgentHandle:
        """Construct a backend-native agent from a neutral spec."""
        ...

    def run_streamed(
        self,
        agent: AgentHandle,
        prompt: str,
        *,
        max_turns: int,
    ) -> AsyncIterator[StreamEvent]:
        """Run the agent against *prompt*, yielding neutral stream events."""
        ...

    async def aclose(self, agent: AgentHandle) -> None:
        """Release any resources held by *agent*."""
        ...
