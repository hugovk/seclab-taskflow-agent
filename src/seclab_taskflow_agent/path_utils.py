# SPDX-FileCopyrightText: 2025 GitHub
# SPDX-License-Identifier: MIT

import platformdirs
import os
from pathlib import Path


def mcp_data_dir(packagename: str, mcpname: str, env_override: str | None) -> Path:
    """
    Create a directory for an MCP to store its data.

    Parameters:
        packagename (str): The name of the package. Used as a subdirectory under the data directory.
        mcpname (str): The name of the MCP server. Used as a subdirectory under the package directory.
        env_override (str | None): The name of an environment variable that, if set, overrides the default data directory location. If None, the default location is used.

    Returns:
        Path: The path to the created data directory for the MCP server.
    """
    if env_override:
        p = os.getenv(env_override)
        if p:
            return Path(p)
    # Use [platformdirs](https://pypi.org/project/platformdirs/) to
    # choose an appropriate location.
    d = platformdirs.user_data_dir(appname="seclab-taskflow-agent", appauthor="GitHubSecurityLab", ensure_exists=True)
    # Each MCP server gets its own sub-directory
    p = Path(d).joinpath(packagename).joinpath(mcpname)
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_dir() -> Path:
    """
    Get the directory path for storing log files for the seclab-taskflow-agent.

    Returns:
        Path: The path to the log directory.
    """
    p = os.getenv("LOG_DIR")
    if not p:
        p = platformdirs.user_log_dir(
            appname="seclab-taskflow-agent", appauthor="GitHubSecurityLab", ensure_exists=True
        )
    return Path(p)


def log_file(filename: str) -> Path:
    """
    Construct the full path to a log file in the user log directory.

    Parameters:
        filename (str): The name of the log file.

    Returns:
        Path: The full path to the log file in the user log directory.
    """
    return log_dir().joinpath(filename)


def log_file_name(filename: str) -> str:
    """
    Construct the full path to a log file in the user log directory.

    Parameters:
        filename (str): The name of the log file.

    Returns:
        str: The full path to the log file in the user log directory.
    """
    return str(log_file(filename))
