# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Capability gating for backend-agnostic taskflow execution.

The runner calls :func:`assert_supported` once per task, after backend
resolution, so a YAML field that the active backend cannot honour
fails immediately with a clear message instead of producing a confusing
error from deep inside the adapter.

Each rejection raises :class:`~.errors.BackendCapabilityError` and names
both the offending field and the backend.
"""

from __future__ import annotations

__all__ = ["assert_supported"]

from typing import Any

from .base import BackendCapabilities
from .errors import BackendCapabilityError


def _reject(backend_name: str, field: str, detail: str = "") -> None:
    suffix = f" ({detail})" if detail else ""
    raise BackendCapabilityError(
        f"Backend {backend_name!r} does not support {field}{suffix}"
    )


def assert_supported(
    caps: BackendCapabilities,
    *,
    agents_count: int,
    api_type: str | None,
    model_settings: dict[str, Any],
    exclude_from_context: bool,
) -> None:
    """Raise if any taskflow option is unsupported by *caps*.

    Args:
        caps: Capability descriptor of the active backend.
        agents_count: Number of personalities resolved for the task. A
            value > 1 implies handoffs.
        api_type: Effective ``api_type`` for the task (``"chat_completions"``
            or ``"responses"``).
        model_settings: Effective per-task model settings dict (after
            engine keys like ``api_type``/``endpoint``/``token`` have
            been stripped).
        exclude_from_context: Whether the task asked the runner to
            exclude tool output from the agent context.
    """
    if agents_count > 1 and not caps.handoffs:
        _reject(caps.name, "agent handoffs", f"{agents_count} personalities configured")

    if api_type == "responses" and not caps.responses_api:
        _reject(caps.name, "api_type='responses'")

    if "temperature" in model_settings and not caps.temperature:
        _reject(caps.name, "model_settings.temperature")

    if "reasoning_effort" in model_settings and not caps.reasoning_effort:
        _reject(caps.name, "model_settings.reasoning_effort")

    if exclude_from_context and not caps.tool_use_behavior_exclude:
        _reject(caps.name, "exclude_from_context=true")
