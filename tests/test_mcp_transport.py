# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for MCP transport utilities."""

import os

from seclab_taskflow_agent.mcp_transport import _filtered_env


class TestFilteredEnv:
    """Tests for _filtered_env environment denylist."""

    def test_no_denylist_copies_env(self, monkeypatch):
        """Without TASKFLOW_ENV_DENYLIST, returns full env copy."""
        monkeypatch.delenv("TASKFLOW_ENV_DENYLIST", raising=False)
        monkeypatch.setenv("TEST_VAR_A", "hello")
        result = _filtered_env()
        assert result["TEST_VAR_A"] == "hello"
        assert result is not os.environ

    def test_denylist_strips_variables(self, monkeypatch):
        """Comma-separated denylist removes matching variables."""
        monkeypatch.setenv("SECRET_TOKEN", "s3cret")
        monkeypatch.setenv("MY_API_KEY", "key123")
        monkeypatch.setenv("SAFE_VAR", "keep")
        monkeypatch.setenv("TASKFLOW_ENV_DENYLIST", "SECRET_TOKEN,MY_API_KEY")
        result = _filtered_env()
        assert "SECRET_TOKEN" not in result
        assert "MY_API_KEY" not in result
        assert result["SAFE_VAR"] == "keep"

    def test_denylist_handles_whitespace(self, monkeypatch):
        """Whitespace around denylist entries is trimmed."""
        monkeypatch.setenv("FOO", "bar")
        monkeypatch.setenv("TASKFLOW_ENV_DENYLIST", " FOO , ")
        result = _filtered_env()
        assert "FOO" not in result

    def test_empty_denylist_copies_env(self, monkeypatch):
        """Empty TASKFLOW_ENV_DENYLIST behaves like unset."""
        monkeypatch.setenv("TASKFLOW_ENV_DENYLIST", "")
        monkeypatch.setenv("KEEP_ME", "yes")
        result = _filtered_env()
        assert result["KEEP_ME"] == "yes"
