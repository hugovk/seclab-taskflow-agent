# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Custom exception classes for seclab-taskflow-agent.

Defines project-specific exception types so that error messages are
encapsulated inside the exception class (satisfying TRY003) and are
therefore discoverable and reusable across the codebase.
"""

from __future__ import annotations

__all__ = [
    "AITokenNotFoundError",
    "ExecutableNotFoundError",
    "InvalidGlobalVariableError",
    "MaxRateLimitReachedError",
    "MCPConnectionTimeoutError",
    "MissingHostPortError",
    "MutuallyExclusiveTaskFieldsError",
    "NoAgentsResolvedError",
    "PersonalityNotFoundError",
    "ProcessThreadTimeoutError",
    "PromptTemplateRenderingError",
    "RequiredEnvVarNotFoundError",
    "ResultTextNotJSONError",
    "ReusableTaskflowNotFoundError",
    "ReusableTaskflowTooManyTasksError",
    "SessionNotFoundError",
    "ShellAndPromptMutuallyExclusiveError",
    "ShellCommandError",
    "TaskModelSettingsTypeError",
    "TemplateRenderingError",
    "TokenEnvVarNotSetError",
    "ToolResultNotJSONError",
    "UnknownModelSettingsError",
    "UnsupportedEndpointError",
    "UnsupportedMCPTransportError",
    "UnsupportedVersionError",
]

from typing import Any

import typer
from openai import APITimeoutError

# ---------------------------------------------------------------------------
# API / token errors
# ---------------------------------------------------------------------------


class UnsupportedEndpointError(ValueError):
    def __init__(self, endpoint: str) -> None:
        super().__init__(f"Unsupported endpoint: {endpoint}")


class AITokenNotFoundError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("AI_API_TOKEN environment variable is not set.")


class TokenEnvVarNotSetError(RuntimeError):
    def __init__(self, token: str) -> None:
        super().__init__(f"Token env var {token!r} is not set")


# ---------------------------------------------------------------------------
# CLI errors
# ---------------------------------------------------------------------------


class InvalidGlobalVariableError(typer.BadParameter):
    def __init__(self, value: str) -> None:
        super().__init__(f"Invalid global variable format: {value!r}. Expected KEY=VALUE.")


# ---------------------------------------------------------------------------
# MCP transport errors
# ---------------------------------------------------------------------------


class UnsupportedMCPTransportError(ValueError):
    def __init__(self, kind: str) -> None:
        super().__init__(f"Unsupported MCP transport: {kind}")


class MissingHostPortError(ValueError):
    def __init__(self, url: str) -> None:
        super().__init__(f"URL must include a host and port: {url}")


class MCPConnectionTimeoutError(TimeoutError):
    def __init__(self, host: str, port: int, timeout: float) -> None:
        super().__init__(f"Could not connect to {host}:{port} after {timeout} seconds")


class ProcessThreadTimeoutError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Process thread did not exit within timeout.")


class ExecutableNotFoundError(FileNotFoundError):
    def __init__(self, command: str) -> None:
        super().__init__(f"Could not resolve path to {command}")


# ---------------------------------------------------------------------------
# Model / config validation errors
# ---------------------------------------------------------------------------


class UnsupportedVersionError(ValueError):
    def __init__(self, version: str, supported: str) -> None:
        super().__init__(f"Unsupported version: {version}. Only version {supported} is supported.")


class MutuallyExclusiveTaskFieldsError(ValueError):
    def __init__(self) -> None:
        super().__init__("shell task ('run') and prompt task ('user_prompt') are mutually exclusive")


# ---------------------------------------------------------------------------
# Runner errors
# ---------------------------------------------------------------------------


class UnknownModelSettingsError(ValueError):
    def __init__(self, config_ref: str, unknown: Any) -> None:
        super().__init__(
            f"Settings section of model_config file {config_ref} contains models not in the model section: {unknown}"
        )


class ReusableTaskflowNotFoundError(ValueError):
    def __init__(self, taskflow_name: str) -> None:
        super().__init__(f"No such reusable taskflow: {taskflow_name}")


class ReusableTaskflowTooManyTasksError(ValueError):
    def __init__(self) -> None:
        super().__init__("Reusable taskflows can only contain 1 task")


class TaskModelSettingsTypeError(ValueError):
    def __init__(self, task_name: str) -> None:
        super().__init__(f"model_settings in task {task_name} needs to be a dictionary")


class ToolResultNotJSONError(ValueError):
    def __init__(self) -> None:
        super().__init__("Tool result is not valid JSON")


class ResultTextNotJSONError(ValueError):
    def __init__(self) -> None:
        super().__init__("Result text is not valid JSON")


class TemplateRenderingError(ValueError):
    def __init__(self, error: Exception) -> None:
        super().__init__(f"Template rendering failed: {error}")


class MaxRateLimitReachedError(APITimeoutError):
    def __init__(self) -> None:
        super().__init__("Max rate limit backoff reached")


class ShellAndPromptMutuallyExclusiveError(ValueError):
    def __init__(self) -> None:
        super().__init__("shell task and prompt task are mutually exclusive!")


class PromptTemplateRenderingError(ValueError):
    def __init__(self, error: Exception) -> None:
        super().__init__(f"Failed to render prompt template: {error}")


class PersonalityNotFoundError(ValueError):
    def __init__(self, agent_name: str) -> None:
        super().__init__(f"No such personality: {agent_name}")


class NoAgentsResolvedError(ValueError):
    def __init__(self) -> None:
        super().__init__(
            "No agents resolved for this task. "
            "Specify a personality with -p or provide an agents list."
        )


# ---------------------------------------------------------------------------
# Session errors
# ---------------------------------------------------------------------------


class SessionNotFoundError(FileNotFoundError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"No session checkpoint found: {session_id}")


# ---------------------------------------------------------------------------
# Shell / process errors
# ---------------------------------------------------------------------------


class ShellCommandError(RuntimeError):
    def __init__(self, cmd: str, stderr: str) -> None:
        super().__init__(f"Command {cmd} failed: {stderr}")


# ---------------------------------------------------------------------------
# Template / environment errors
# ---------------------------------------------------------------------------


class RequiredEnvVarNotFoundError(LookupError):
    def __init__(self, var_name: str) -> None:
        super().__init__(f"Required environment variable {var_name} not found!")
