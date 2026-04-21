# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Stream-driving helpers for the runner.

This module owns the inner loop that consumes events from a backend
adapter (`TextDelta` / `ToolEnd`), renders text deltas to the user, and
bridges Copilot-side tool events into the run-hook callbacks that the
runner uses to capture MCP results for ``repeat_prompt`` and session
checkpointing.

Extracted from ``runner.py`` so the rate-limit/retry loop and the
backend-event translation are independently readable and testable.
"""

from __future__ import annotations

__all__ = ["bridge_copilot_tool_event", "drive_backend_stream"]

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any

from .render_utils import render_model_output
from .sdk import TextDelta, ToolEnd
from .sdk.errors import BackendRateLimitError, BackendTimeoutError


async def bridge_copilot_tool_event(event: ToolEnd, run_hooks: Any) -> None:
    """Forward a Copilot ``ToolEnd`` into the openai-agents-style hooks.

    The runner captures MCP tool output via ``run_hooks.on_tool_end``,
    which the openai-agents path drives natively. The Copilot adapter
    surfaces tool completions as ``ToolEnd`` events instead, so we
    invoke the same hooks here with:

    * a ``SimpleNamespace(name=...)`` placeholder in lieu of the
      openai-agents ``Tool`` object — the hooks only read ``.name``.
    * a ``json.dumps({"text": ...})`` envelope around the result text,
      matching the wire format openai-agents uses when serialising MCP
      ``TextContent`` lists. ``_build_prompts_to_run`` in the runner
      depends on that exact envelope shape, so both backends produce
      identical entries in ``last_mcp_tool_results``.
    """
    if run_hooks is None:
        return
    fake_tool = SimpleNamespace(name=event.tool_name)
    payload = json.dumps({"text": event.text})
    await run_hooks.on_tool_start(None, None, fake_tool)
    await run_hooks.on_tool_end(None, None, fake_tool, payload)


async def drive_backend_stream(
    *,
    backend_impl: Any,
    agent_handle: Any,
    prompt: str,
    max_turns: int,
    run_hooks: Any,
    async_task: bool,
    task_id: str,
    max_api_retry: int,
    initial_rate_limit_backoff: int,
    max_rate_limit_backoff: int,
) -> None:
    """Run the backend's event stream to completion with retry/backoff.

    Renders ``TextDelta`` events to stdout, forwards ``ToolEnd`` events
    to the run-hook bridge, retries up to *max_api_retry* times on
    :class:`BackendTimeoutError`, and applies exponential backoff up to
    *max_rate_limit_backoff* seconds on :class:`BackendRateLimitError`
    before giving up with a :class:`BackendTimeoutError`.
    """
    max_retry = max_api_retry
    rate_limit_backoff = initial_rate_limit_backoff
    last_rate_limit_exc: BackendRateLimitError | None = None

    while rate_limit_backoff:
        try:
            async for event in backend_impl.run_streamed(
                agent_handle, prompt, max_turns=max_turns
            ):
                if isinstance(event, TextDelta):
                    await render_model_output(
                        event.text, async_task=async_task, task_id=task_id
                    )
                elif isinstance(event, ToolEnd):
                    await bridge_copilot_tool_event(event, run_hooks)
            await render_model_output("\n\n", async_task=async_task, task_id=task_id)
            return
        except BackendTimeoutError:
            if not max_retry:
                logging.exception("Max retries for BackendTimeoutError reached")
                raise
            max_retry -= 1
        except BackendRateLimitError as exc:
            last_rate_limit_exc = exc
            if rate_limit_backoff == max_rate_limit_backoff:
                raise BackendTimeoutError("Max rate limit backoff reached") from exc
            if rate_limit_backoff > max_rate_limit_backoff:
                rate_limit_backoff = max_rate_limit_backoff
            else:
                rate_limit_backoff += rate_limit_backoff
            logging.exception(f"Hit rate limit ... holding for {rate_limit_backoff}")
            await asyncio.sleep(rate_limit_backoff)

    if last_rate_limit_exc is not None:  # pragma: no cover - loop always returns/raises above
        raise BackendTimeoutError("Rate limit backoff exhausted") from last_rate_limit_exc
