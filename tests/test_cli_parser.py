# SPDX-FileCopyrightText: 2025 GitHub
# SPDX-License-Identifier: MIT

"""
Test CLI global variable parsing.
"""

import pytest

from seclab_taskflow_agent.available_tools import AvailableTools


class TestCliGlobals:
    """Test CLI global variable parsing."""

    def test_parse_single_global(self):
        """Test parsing a single global variable from command line."""
        from seclab_taskflow_agent.__main__ import parse_prompt_args

        available_tools = AvailableTools()

        p, t, l, cli_globals, user_prompt, _ = parse_prompt_args(available_tools, "-t example -g fruit=apples")

        assert t == "example"
        assert cli_globals == {"fruit": "apples"}
        assert p is None
        assert l is False

    def test_parse_multiple_globals(self):
        """Test parsing multiple global variables from command line."""
        from seclab_taskflow_agent.__main__ import parse_prompt_args

        available_tools = AvailableTools()

        p, t, l, cli_globals, user_prompt, _ = parse_prompt_args(
            available_tools, "-t example -g fruit=apples -g color=red"
        )

        assert t == "example"
        assert cli_globals == {"fruit": "apples", "color": "red"}
        assert p is None
        assert l is False

    def test_parse_global_with_spaces(self):
        """Test parsing global variables with spaces in values."""
        from seclab_taskflow_agent.__main__ import parse_prompt_args

        available_tools = AvailableTools()

        p, t, l, cli_globals, user_prompt, _ = parse_prompt_args(available_tools, "-t example -g message=hello world")

        assert t == "example"
        # "world" becomes part of the prompt, not the value
        assert cli_globals == {"message": "hello"}
        assert "world" in user_prompt

    def test_parse_global_with_equals_in_value(self):
        """Test parsing global variables with equals sign in value."""
        from seclab_taskflow_agent.__main__ import parse_prompt_args

        available_tools = AvailableTools()

        p, t, l, cli_globals, user_prompt, _ = parse_prompt_args(available_tools, "-t example -g equation=x=5")

        assert t == "example"
        assert cli_globals == {"equation": "x=5"}

    def test_globals_in_taskflow_file(self):
        """Test that globals can be read from taskflow file."""
        available_tools = AvailableTools()

        taskflow = available_tools.get_taskflow("tests.data.test_globals_taskflow")
        assert "globals" in taskflow
        assert taskflow["globals"]["test_var"] == "default_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
