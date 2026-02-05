# SPDX-FileCopyrightText: 2025 GitHub
# SPDX-License-Identifier: MIT

# https://openai.github.io/openai-agents-python/agents/
import logging
import os
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from agents import (
    Agent,
    AgentHooks,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    RunHooks,
    Runner,
    TContext,
    Tool,
    result,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from agents.agent import FunctionToolResult, ModelSettings, ToolsToFinalOutputResult
from agents.run import DEFAULT_MAX_TURNS, RunHooks
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI

from .capi import AI_API_ENDPOINT_ENUM, COPILOT_INTEGRATION_ID, get_AI_endpoint, get_AI_token

# grab our secrets from .env, this must be in .gitignore
load_dotenv(find_dotenv(usecwd=True))

api_endpoint = get_AI_endpoint()
match urlparse(api_endpoint).netloc:
    case AI_API_ENDPOINT_ENUM.AI_API_GITHUBCOPILOT:
        default_model = "gpt-4o"
    case AI_API_ENDPOINT_ENUM.AI_API_MODELS_GITHUB:
        default_model = "openai/gpt-4o"
    case AI_API_ENDPOINT_ENUM.AI_API_OPENAI:
        default_model = "gpt-4o"
    case _:
        raise ValueError(
            f"Unsupported Model Endpoint: {api_endpoint}\n"
            f"Supported endpoints: {[e.to_url() for e in AI_API_ENDPOINT_ENUM]}"
        )

DEFAULT_MODEL = os.getenv("COPILOT_DEFAULT_MODEL", default=default_model)


# Run hooks monitor the entire lifetime of a runner, including across any Agent handoffs
class TaskRunHooks(RunHooks):
    def __init__(
        self,
        on_agent_start: Callable | None = None,
        on_agent_end: Callable | None = None,
        on_tool_start: Callable | None = None,
        on_tool_end: Callable | None = None,
    ):
        self._on_agent_start = on_agent_start
        self._on_agent_end = on_agent_end
        self._on_tool_start = on_tool_start
        self._on_tool_end = on_tool_end

    async def on_agent_start(self, context: RunContextWrapper[TContext], agent: Agent[TContext]) -> None:
        logging.debug(f"TaskRunHooks on_agent_start: {agent.name}")
        if self._on_agent_start:
            await self._on_agent_start(context, agent)

    async def on_agent_end(self, context: RunContextWrapper[TContext], agent: Agent[TContext], output: Any) -> None:
        logging.debug(f"TaskRunHooks on_agent_end: {agent.name}")
        if self._on_agent_end:
            await self._on_agent_end(context, agent, output)

    async def on_tool_start(self, context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool) -> None:
        logging.debug(f"TaskRunHooks on_tool_start: {tool.name}")
        if self._on_tool_start:
            await self._on_tool_start(context, agent, tool)

    async def on_tool_end(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool, result: str
    ) -> None:
        logging.debug(f"TaskRunHooks on_tool_end: {tool.name} ")
        if self._on_tool_end:
            await self._on_tool_end(context, agent, tool, result)


# Agent hooks monitor the lifetime of a single agent, not across any Agent handoffs
class TaskAgentHooks(AgentHooks):
    def __init__(
        self,
        on_handoff: Callable | None = None,
        on_start: Callable | None = None,
        on_end: Callable | None = None,
        on_tool_start: Callable | None = None,
        on_tool_end: Callable | None = None,
    ):
        self._on_handoff = on_handoff
        self._on_start = on_start
        self._on_end = on_end
        self._on_tool_start = on_tool_start
        self._on_tool_end = on_tool_end

    async def on_handoff(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], source: Agent[TContext]
    ) -> None:
        logging.debug(f"TaskAgentHooks on_handoff: {source.name} -> {agent.name}")
        if self._on_handoff:
            await self._on_handoff(context, agent, source)

    async def on_start(self, context: RunContextWrapper[TContext], agent: Agent[TContext]) -> None:
        logging.debug(f"TaskAgentHooks on_start: {agent.name}")
        if self._on_start:
            await self._on_start(context, agent)

    async def on_end(self, context: RunContextWrapper[TContext], agent: Agent[TContext], output: Any) -> None:
        logging.debug(f"TaskAgentHooks on_end: {agent.name}")
        if self._on_end:
            await self._on_end(context, agent, output)

    async def on_tool_start(self, context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool) -> None:
        logging.debug(f"TaskAgentHooks on_tool_start: {tool.name}")
        if self._on_tool_start:
            await self._on_tool_start(context, agent, tool)

    async def on_tool_end(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool, result: str
    ) -> None:
        logging.debug(f"TaskAgentHooks on_tool_end: {tool.name}")
        if self._on_tool_end:
            await self._on_tool_end(context, agent, tool, result)


class TaskAgent:
    def __init__(
        self,
        name: str = "TaskAgent",
        instructions: str = "",
        handoffs: list = [],
        exclude_from_context: bool = False,
        mcp_servers: dict = [],
        model: str = DEFAULT_MODEL,
        model_settings: ModelSettings | None = None,
        run_hooks: TaskRunHooks | None = None,
        agent_hooks: TaskAgentHooks | None = None,
    ):
        client = AsyncOpenAI(
            base_url=api_endpoint,
            api_key=get_AI_token(),
            default_headers={"Copilot-Integration-Id": COPILOT_INTEGRATION_ID},
        )
        set_default_openai_client(client)
        # CAPI does not yet support the Responses API: https://github.com/github/copilot-api/issues/11185
        # as such we are implementing on chat completions for now
        set_default_openai_api("chat_completions")
        set_tracing_disabled(True)
        self.run_hooks = run_hooks or TaskRunHooks()
        # useful agent patterns:
        # openai/openai-agents-python/blob/main/examples/agent_patterns

        # when we want to exclude tool results from context, we receive results here instead of sending to LLM
        def _ToolsToFinalOutputFunction(
            context: RunContextWrapper[TContext], results: list[FunctionToolResult]
        ) -> ToolsToFinalOutputResult:
            return ToolsToFinalOutputResult(True, "Excluding tool results from LLM context")

        self.agent = Agent(
            name=name,
            instructions=instructions,
            tool_use_behavior=_ToolsToFinalOutputFunction if exclude_from_context else "run_llm_again",
            model=OpenAIChatCompletionsModel(model=model, openai_client=client),
            handoffs=handoffs,
            mcp_servers=mcp_servers,
            model_settings=model_settings or ModelSettings(),
            hooks=agent_hooks or TaskAgentHooks(),
        )

    async def run(self, prompt: str, max_turns: int = DEFAULT_MAX_TURNS) -> result.RunResult:
        return await Runner.run(starting_agent=self.agent, input=prompt, max_turns=max_turns, hooks=self.run_hooks)

    def run_streamed(self, prompt: str, max_turns: int = DEFAULT_MAX_TURNS) -> result.RunResultStreaming:
        return Runner.run_streamed(starting_agent=self.agent, input=prompt, max_turns=max_turns, hooks=self.run_hooks)
