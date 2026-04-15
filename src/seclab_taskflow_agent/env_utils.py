# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Environment variable utilities for taskflow execution."""

import os
from typing import Any

import jinja2

__all__ = ["TmpEnv", "swap_env"]



def swap_env(s: str, context: dict[str, Any] | None = None) -> str:
    """Replace {{ env('VAR') }} and {{ globals.X }} patterns in string.

    Args:
        s: String potentially containing templates
        context: Optional template context (e.g. {'globals': {...}})

    Returns:
        String with templates replaced

    Raises:
        LookupError: If required env var not found
    """
    # Quick check if templating needed
    if '{{' not in s:
        return s

    try:
        # Import here to avoid circular dependency
        from .template_utils import create_jinja_environment
        from .available_tools import AvailableTools

        available_tools = AvailableTools()
        jinja_env = create_jinja_environment(available_tools)
        template = jinja_env.from_string(s)
        return template.render(**(context or {}))
    except jinja2.UndefinedError as e:
        # Convert Jinja undefined to LookupError for compatibility
        raise LookupError(str(e))
    except jinja2.TemplateError:
        # Not a template or failed to render, return as-is
        return s


class TmpEnv:
    """Context manager that temporarily sets environment variables."""

    def __init__(self, env: dict[str, str],
                 context: dict[str, Any] | None = None) -> None:
        self.env = dict(env)
        self.context = context
        self.restore_env = dict(os.environ)

    def __enter__(self) -> None:
        for k, v in self.env.items():
            os.environ[k] = swap_env(v, self.context)

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any | None) -> None:
        for k, v in self.env.items():
            del os.environ[k]
            if k in self.restore_env:
                os.environ[k] = self.restore_env[k]
