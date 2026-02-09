# SPDX-FileCopyrightText: 2025 GitHub
# SPDX-License-Identifier: MIT

import os
import jinja2



def swap_env(s: str) -> str:
    """Replace {{ env('VAR') }} patterns in string with environment values.

    Args:
        s: String potentially containing env templates

    Returns:
        String with env templates replaced

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
        return template.render()
    except jinja2.UndefinedError as e:
        # Convert Jinja undefined to LookupError for compatibility
        raise LookupError(str(e))
    except jinja2.TemplateError:
        # Not a template or failed to render, return as-is
        return s


class TmpEnv:
    def __init__(self, env):
        self.env = dict(env)
        self.restore_env = dict(os.environ)

    def __enter__(self):
        for k, v in self.env.items():
            os.environ[k] = swap_env(v)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for k, v in self.env.items():
            del os.environ[k]
            if k in self.restore_env:
                os.environ[k] = self.restore_env[k]
