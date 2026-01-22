# SPDX-FileCopyrightText: 2025 GitHub
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
        personality000 = available_tools.get_personality("tests.data.test_yaml_parser_personality000")

        assert personality000["seclab-taskflow-agent"]["version"] == 1
        assert personality000["seclab-taskflow-agent"]["filetype"] == "personality"
        assert personality000["personality"] == "You are a helpful assistant.\n"
        assert personality000["task"] == "Answer any question.\n"


class TestRealTaskflowFiles:
    """Test parsing of actual taskflow files in the project."""

    def test_parse_example_taskflows(self):
        """Test parsing example taskflow files."""
        # this test uses the actual taskflows in the project
        available_tools = AvailableTools()

        # check that example.yaml is parsed correctly
        example_task_flow = available_tools.get_taskflow("examples.taskflows.example")
        assert "taskflow" in example_task_flow
        assert isinstance(example_task_flow["taskflow"], list)
        assert len(example_task_flow["taskflow"]) == 4  # 4 tasks in taskflow
        assert example_task_flow["taskflow"][0]["task"]["max_steps"] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
