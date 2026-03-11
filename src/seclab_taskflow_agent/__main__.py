# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Entry point for ``python -m seclab_taskflow_agent``.

This module serves as the package entry point. The actual implementation
is split across focused modules:

- :mod:`~seclab_taskflow_agent.cli` — CLI argument parsing (Typer)
- :mod:`~seclab_taskflow_agent.runner` — Taskflow execution engine
- :mod:`~seclab_taskflow_agent.mcp_lifecycle` — MCP server lifecycle
- :mod:`~seclab_taskflow_agent.models` — Pydantic grammar models
- :mod:`~seclab_taskflow_agent.agent` — Agent wrapper classes
"""

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

# Re-export for backwards compatibility — some tests import from __main__
from .prompt_parser import parse_prompt_args  # noqa: E402, F401
from .runner import deploy_task_agents, run_main  # noqa: E402, F401

if __name__ == "__main__":
    from .cli import app

    app()
