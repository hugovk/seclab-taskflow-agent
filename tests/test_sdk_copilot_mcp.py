# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``sdk.copilot_sdk.mcp``."""

from __future__ import annotations

from seclab_taskflow_agent.sdk.base import MCPServerSpec
from seclab_taskflow_agent.sdk.copilot_sdk.mcp import build_mcp_config


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
                "timeout": 2.5,
            },
        )
    ]
    assert build_mcp_config(specs) == {
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
                "timeout": 10.0,
            },
        )
    ]
    assert build_mcp_config(specs) == {
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
    out = build_mcp_config(specs)
    assert out["gh"]["type"] == "http"
    assert out["gh"]["url"] == "https://api.example.com/mcp"


def test_streamable_spec_without_url_is_skipped():
    assert build_mcp_config([MCPServerSpec(name="x", kind="streamable", params={})]) == {}


def test_stdio_minimal_defaults():
    specs = [MCPServerSpec(name="m", kind="stdio", params={"command": "x"})]
    assert build_mcp_config(specs) == {
        "m": {"type": "stdio", "command": "x", "args": [], "tools": ["*"]}
    }
