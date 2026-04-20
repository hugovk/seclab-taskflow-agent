# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``sdk.copilot_sdk.mcp``."""

from __future__ import annotations

from seclab_taskflow_agent.sdk.base import MCPServerSpec
from seclab_taskflow_agent.sdk.copilot_sdk.mcp import materialise_mcp_servers


def test_stdio_spec_translated_to_subprocess_config():
    specs = [
        MCPServerSpec(
            name="files",
            kind="stdio",
            params={
                "command": "/usr/bin/mcp",
                "args": ["--root", "/tmp"],
                "env": {"FOO": "1"},
                "cwd": "/work",
            },
            client_session_timeout=2.5,
        )
    ]
    result = materialise_mcp_servers(specs)
    assert result == {
        "files": {
            "type": "stdio",
            "command": "/usr/bin/mcp",
            "args": ["--root", "/tmp"],
            "env": {"FOO": "1"},
            "cwd": "/work",
            "tools": ["*"],
            "timeout": 2500,
        }
    }


def test_sse_spec_translated_to_remote_config():
    specs = [
        MCPServerSpec(
            name="docs",
            kind="sse",
            params={
                "url": "https://example.com/mcp",
                "headers": {"Authorization": "Bearer x"},
            },
            client_session_timeout=10.0,
        )
    ]
    result = materialise_mcp_servers(specs)
    assert result == {
        "docs": {
            "type": "sse",
            "url": "https://example.com/mcp",
            "headers": {"Authorization": "Bearer x"},
            "tools": ["*"],
            "timeout": 10000,
        }
    }


def test_streamable_spec_uses_http_type():
    specs = [
        MCPServerSpec(
            name="gh",
            kind="streamable",
            params={"url": "https://api.example.com/mcp"},
        )
    ]
    result = materialise_mcp_servers(specs)
    assert result["gh"]["type"] == "http"
    assert result["gh"]["url"] == "https://api.example.com/mcp"


def test_streamable_spec_without_url_is_skipped():
    specs = [MCPServerSpec(name="inproc", kind="streamable", params={})]
    assert materialise_mcp_servers(specs) == {}


def test_stdio_minimal_defaults():
    specs = [MCPServerSpec(name="m", kind="stdio", params={"command": "x"})]
    result = materialise_mcp_servers(specs)
    assert result["m"] == {"type": "stdio", "command": "x", "args": [], "tools": ["*"]}
