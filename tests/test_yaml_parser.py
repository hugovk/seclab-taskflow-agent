# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""
Basic tests for YAML parsing functionality in the taskflow agent.

Simple parsing + parsing of example taskflows.
"""

import pytest

from seclab_taskflow_agent.available_tools import AvailableTools, AvailableToolType, BadToolNameError


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


class TestListResources:
    """Tests for AvailableTools.list_resources()."""

    def test_list_resources_returns_all_types(self):
        """list_resources() returns a mapping with all AvailableToolType keys."""
        at = AvailableTools()
        result = at.list_resources()
        assert set(result.keys()) == set(AvailableToolType)

    def test_list_resources_taskflows_not_empty(self):
        """list_resources() finds at least the example taskflows."""
        at = AvailableTools()
        result = at.list_resources()
        taskflows = result[AvailableToolType.Taskflow]
        assert len(taskflows) > 0
        assert "examples.taskflows.example" in taskflows

    def test_list_resources_personalities_not_empty(self):
        """list_resources() finds at least the built-in personalities."""
        at = AvailableTools()
        result = at.list_resources()
        personalities = result[AvailableToolType.Personality]
        assert len(personalities) > 0
        assert "seclab_taskflow_agent.personalities.assistant" in personalities

    def test_list_resources_single_type(self):
        """list_resources(tooltype=…) returns only that type."""
        at = AvailableTools()
        result = at.list_resources(AvailableToolType.Taskflow)
        assert set(result.keys()) == {AvailableToolType.Taskflow}
        assert len(result[AvailableToolType.Taskflow]) > 0

    def test_list_resources_sorted(self):
        """list_resources() returns resource names in sorted order."""
        at = AvailableTools()
        result = at.list_resources()
        for tt, names in result.items():
            assert names == sorted(names), f"{tt} resources are not sorted"

    def test_list_resources_dotted_name_format(self):
        """All returned resource names follow the package.subdir.stem format."""
        at = AvailableTools()
        result = at.list_resources()
        for names in result.values():
            for name in names:
                parts = name.split(".")
                assert len(parts) >= 3, f"Expected at least 3 parts in {name!r}"

    def test_not_found_error_includes_available_hint(self):
        """BadToolNameError for missing taskflow includes available taskflow list."""
        at = AvailableTools()
        with pytest.raises(BadToolNameError) as exc_info:
            at.get_taskflow("examples.taskflows.this_does_not_exist")
        message = str(exc_info.value)
        assert "Available taskflows" in message
        assert "examples.taskflows.example" in message

    def test_not_found_personality_includes_available_hint(self):
        """BadToolNameError for missing personality includes available personality list."""
        at = AvailableTools()
        with pytest.raises(BadToolNameError) as exc_info:
            at.get_personality("seclab_taskflow_agent.personalities.no_such_personality")
        message = str(exc_info.value)
        assert "Available personalities" in message
        assert "seclab_taskflow_agent.personalities.assistant" in message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
