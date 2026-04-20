# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Backend-neutral SDK abstraction for the taskflow runner.

The runner consumes only this package; concrete adapters live in
``sdk.openai_agents`` and ``sdk.copilot_sdk`` and are loaded lazily so
that optional dependencies (the Copilot SDK in particular) are only
imported when actually requested.
"""

from __future__ import annotations

__all__ = [
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
    "available_backends",
    "get_backend_capabilities",
    "register_backend_capabilities",
    "resolve_backend_name",
]

import os
from typing import get_args

from .base import (
    AgentSpec,
    BackendCapabilities,
    BackendName,
    MCPServerSpec,
    RunResult,
    StreamEvent,
    TextDelta,
    ToolEnd,
    ToolSpec,
    ToolStart,
)

_ENV_VAR = "SECLAB_TASKFLOW_BACKEND"

_CAPABILITIES: dict[str, BackendCapabilities] = {}


def register_backend_capabilities(caps: BackendCapabilities) -> None:
    """Register a backend's capability descriptor.

    Adapters call this at import time so consumers can introspect
    ``available_backends()`` and ``get_backend_capabilities()`` without
    triggering the adapter's own optional dependencies.
    """
    if caps.name not in get_args(BackendName):
        raise ValueError(f"Unknown backend name: {caps.name!r}")
    _CAPABILITIES[caps.name] = caps


def available_backends() -> tuple[str, ...]:
    """Return the names of backends whose capabilities have been registered."""
    return tuple(sorted(_CAPABILITIES))


def get_backend_capabilities(name: str) -> BackendCapabilities:
    """Return the capability descriptor for the named backend."""
    try:
        return _CAPABILITIES[name]
    except KeyError as exc:
        raise ValueError(
            f"Backend {name!r} is not registered. Known backends: {available_backends()}"
        ) from exc


def resolve_backend_name(
    *,
    explicit: str | None = None,
    endpoint: str | None = None,
) -> str:
    """Pick the backend to use for a run.

    Resolution order, highest precedence first:

    1. ``explicit`` argument (typically ``ModelConfigDocument.backend``).
    2. ``SECLAB_TASKFLOW_BACKEND`` environment variable.
    3. Endpoint-based auto-default — ``api.githubcopilot.com`` prefers
       ``copilot_sdk`` when its capabilities have been registered.
    4. ``openai_agents`` as the safe default.

    The chosen name is validated against the registered backends.
    """
    candidate = explicit or os.getenv(_ENV_VAR) or _auto_default(endpoint)
    if candidate not in _CAPABILITIES:
        raise ValueError(
            f"Backend {candidate!r} is not available. "
            f"Known backends: {available_backends()}"
        )
    return candidate


def _auto_default(endpoint: str | None) -> str:
    if endpoint and "api.githubcopilot.com" in endpoint and "copilot_sdk" in _CAPABILITIES:
        return "copilot_sdk"
    return "openai_agents"
