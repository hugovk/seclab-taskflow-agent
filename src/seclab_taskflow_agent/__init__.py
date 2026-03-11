# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""SecLab Taskflow Agent — secure and automated workflow execution.

This package provides the engine for running declarative YAML-based taskflows
that orchestrate AI agents with MCP (Model Context Protocol) tool servers
for security analysis, code auditing, and vulnerability triage.

Architecture
~~~~~~~~~~~~
- :mod:`~seclab_taskflow_agent.models` — Pydantic grammar models
- :mod:`~seclab_taskflow_agent.cli` — CLI entry point (Typer)
- :mod:`~seclab_taskflow_agent.runner` — Taskflow execution engine
- :mod:`~seclab_taskflow_agent.agent` — Agent wrapper classes
- :mod:`~seclab_taskflow_agent.mcp_lifecycle` — MCP server lifecycle
- :mod:`~seclab_taskflow_agent.mcp_utils` — MCP utilities
- :mod:`~seclab_taskflow_agent.template_utils` — Jinja2 template rendering
- :mod:`~seclab_taskflow_agent.available_tools` — YAML resource loader
"""
