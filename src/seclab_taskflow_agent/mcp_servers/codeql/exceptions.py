# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Custom exception classes for the CodeQL MCP server."""

from __future__ import annotations

__all__ = [
    "DatabaseNotFoundError",
    "LegacyServerNotSupportedError",
    "NoActiveConnectionError",
    "NoActiveDatabaseError",
    "NonAbsoluteURIError",
    "NotFileURIError",
    "QueryRunError",
    "QuickEvalTargetNotFoundError",
    "UnsupportedLanguageError",
    "UnsupportedOutputFormatError",
    "UnsupportedQueryError",
]


class NoActiveDatabaseError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("No Active Database")


class NoActiveConnectionError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("No Active Connection")


class LegacyServerNotSupportedError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Legacy server not supported!")


class NonAbsoluteURIError(ValueError):
    def __init__(self) -> None:
        super().__init__("URI path should be formatted as absolute")


class NotFileURIError(ValueError):
    def __init__(self, uri: str) -> None:
        super().__init__(f"Not a file:// uri: {uri}")


class QuickEvalTargetNotFoundError(ValueError):
    def __init__(self, target: str) -> None:
        super().__init__(f"Could not resolve quick eval target for {target}")


class UnsupportedOutputFormatError(ValueError):
    def __init__(self, fmt: str) -> None:
        super().__init__(f"Unsupported output format {fmt}")


class QueryRunError(RuntimeError):
    def __init__(self, error: Exception) -> None:
        super().__init__(f"Error in run_query: {error}")


class UnsupportedLanguageError(RuntimeError):
    def __init__(self, language: str) -> None:
        super().__init__(f"Error: Language `{language}` not supported!")


class UnsupportedQueryError(RuntimeError):
    def __init__(self, query: str, language: str) -> None:
        super().__init__(f"Error: query `{query}` not supported for `{language}`!")


class DatabaseNotFoundError(RuntimeError):
    def __init__(self, path: str) -> None:
        super().__init__(f"Error: Database not found at {path}!")
