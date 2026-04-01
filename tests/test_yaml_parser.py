# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""
Basic tests for YAML parsing functionality in the taskflow agent.

Simple parsing + parsing of example taskflows.
"""

import pytest

from seclab_taskflow_agent.available_tools import AvailableTools


class TestYamlParser:
    """Test suite for YamlParser class."""

    def test_yaml_parser_basic_functionality(self):
        """Test basic YAML parsing functionality."""
        available_tools = AvailableTools()
        personality = available_tools.get_personality(
            "tests.data.test_yaml_parser_personality000")

        assert personality.header.version == "1.0"
        assert personality.header.filetype == "personality"
        assert personality.personality == "You are a helpful assistant.\n"
        assert personality.task == "Answer any question.\n"

    def test_version_integer_format(self):
        """Test that integer version format is accepted."""
        available_tools = AvailableTools()
        personality = available_tools.get_personality(
            "tests.data.test_version_integer")

        # Version is normalized to "1.0" by the model
        assert personality.header.version == "1.0"
        assert personality.header.filetype == "personality"
        assert personality.personality == "Test personality with integer version.\n"

    def test_version_float_format(self):
        """Test that float version format is accepted."""
        available_tools = AvailableTools()
        personality = available_tools.get_personality(
            "tests.data.test_version_float")

        # Version is normalized to "1.0" by the model
        assert personality.header.version == "1.0"
        assert personality.header.filetype == "personality"
        assert personality.personality == "Test personality with float version.\n"

class TestRealTaskflowFiles:
    """Test parsing of actual taskflow files in the project."""

    def test_parse_example_taskflows(self):
        """Test parsing example taskflow files."""
        available_tools = AvailableTools()

        # check that example.yaml is parsed correctly
        example = available_tools.get_taskflow("examples.taskflows.example")
        assert example.taskflow is not None
        assert isinstance(example.taskflow, list)
        assert len(example.taskflow) == 4  # 4 tasks in taskflow
        assert example.taskflow[0].task.max_steps == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
