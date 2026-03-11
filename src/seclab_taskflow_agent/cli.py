# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Command-line interface for the seclab-taskflow-agent.

Provides the Typer-based CLI entry point, replacing the previous argparse
implementation. Supports personality mode (-p), taskflow mode (-t),
model listing (-l), and global variables (-g KEY=VALUE).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Annotated

import typer

from .available_tools import AvailableTools
from .banner import get_banner
from .capi import get_AI_token, list_tool_call_models
from .path_utils import log_file_name

app = typer.Typer(
    name="seclab-taskflow-agent",
    help="SecLab Taskflow Agent — secure and automated workflow execution.",
    add_completion=False,
    no_args_is_help=True,
)


def _parse_global(value: str) -> tuple[str, str]:
    """Parse a ``KEY=VALUE`` string into a (key, value) pair."""
    if "=" not in value:
        raise typer.BadParameter(f"Invalid global variable format: {value!r}. Expected KEY=VALUE.")
    key, _, val = value.partition("=")
    return key.strip(), val.strip()


def _setup_logging() -> None:
    """Configure root logger: file (DEBUG) + console (ERROR)."""
    import os
    from logging.handlers import RotatingFileHandler

    root = logging.getLogger("")
    root.setLevel(logging.NOTSET)

    file_handler = RotatingFileHandler(
        log_file_name("task_agent.log"), maxBytes=10 * 1024 * 1024, backupCount=10
    )
    file_handler.setLevel(os.getenv("TASK_AGENT_LOGLEVEL", "DEBUG"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console_handler)


@app.command()
def main(
    personality: Annotated[
        str | None,
        typer.Option("-p", "--personality", help="Personality module path (mutually exclusive with -t)."),
    ] = None,
    taskflow: Annotated[
        str | None,
        typer.Option("-t", "--taskflow", help="Taskflow module path (mutually exclusive with -p)."),
    ] = None,
    list_models: Annotated[
        bool,
        typer.Option("-l", "--list-models", help="List available tool-call models and exit."),
    ] = False,
    globals_: Annotated[
        list[str] | None,
        typer.Option("-g", "--global", help="Global variable as KEY=VALUE. Repeatable."),
    ] = None,
    prompt: Annotated[
        list[str] | None,
        typer.Argument(help="Remaining prompt text."),
    ] = None,
) -> None:
    """Run a taskflow or personality-based agent session."""
    # Validate mutual exclusivity
    specified = sum(bool(x) for x in [personality, taskflow, list_models])
    if specified > 1:
        typer.echo("Error: -p, -t, and -l are mutually exclusive.", err=True)
        raise typer.Exit(code=1)

    _setup_logging()

    available_tools = AvailableTools()

    # List models mode
    if list_models:
        tool_models = list_tool_call_models(get_AI_token())
        for model in tool_models:
            typer.echo(model)
        raise typer.Exit()

    if personality is None and taskflow is None:
        typer.echo("Error: one of -p or -t is required.", err=True)
        raise typer.Exit(code=1)

    # Parse global variables
    cli_globals: dict[str, str] = {}
    for g in globals_ or []:
        key, val = _parse_global(g)
        cli_globals[key] = val

    user_prompt = " ".join(prompt) if prompt else ""

    typer.echo(get_banner())

    from .runner import run_main

    asyncio.run(
        run_main(available_tools, personality, taskflow, cli_globals, user_prompt),
        debug=os.getenv("TASK_AGENT_LOGLEVEL", "").upper() == "DEBUG",
    )


# ---------------------------------------------------------------------------
# Legacy compatibility shim — implementation moved to prompt_parser.py
# ---------------------------------------------------------------------------

from .prompt_parser import parse_prompt_args  # noqa: F401, E402
