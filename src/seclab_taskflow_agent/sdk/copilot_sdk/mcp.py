# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Translate neutral :class:`MCPServerSpec` values into Copilot SDK MCP configs."""

from __future__ import annotations

__all__ = ["materialise_mcp_servers"]

from typing import Any

from ..base import MCPServerSpec


def _stdio_config(spec: MCPServerSpec) -> dict[str, Any]:
    p = spec.params
    config: dict[str, Any] = {
        "type": "stdio",
        "command": p.get("command", ""),
        "args": list(p.get("args") or []),
        "tools": ["*"],  # session-level excluded_tools handles filtering
    }
    if p.get("env"):
        config["env"] = dict(p["env"])
    if p.get("cwd"):
        config["cwd"] = p["cwd"]
    if spec.client_session_timeout:
        config["timeout"] = int(spec.client_session_timeout * 1000)
    return config


def _remote_config(spec: MCPServerSpec) -> dict[str, Any]:
    p = spec.params
    kind = "sse" if spec.kind == "sse" else "http"
    config: dict[str, Any] = {
        "type": kind,
        "url": p.get("url", ""),
        "tools": ["*"],
    }
    if p.get("headers"):
        config["headers"] = dict(p["headers"])
    timeout_s = spec.client_session_timeout or p.get("timeout") or 0
    if timeout_s:
        config["timeout"] = int(timeout_s * 1000)
    return config


def materialise_mcp_servers(specs: list[MCPServerSpec]) -> dict[str, dict[str, Any]]:
    """Return a ``mcp_servers`` mapping suitable for ``create_session``.

    Streamable specs without a ``url`` (the runner is hosting the
    transport itself in-process) are skipped — the SDK cannot consume
    an in-process transport. The capability gate is expected to reject
    such configurations before this is reached.
    """
    out: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec.kind == "stdio":
            out[spec.name] = _stdio_config(spec)
        elif spec.kind in ("sse", "streamable"):
            if not spec.params.get("url"):
                continue
            out[spec.name] = _remote_config(spec)
    return out
