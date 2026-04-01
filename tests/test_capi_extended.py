# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Extended tests for capi module."""

from __future__ import annotations

from seclab_taskflow_agent.capi import AI_API_ENDPOINT_ENUM, supports_tool_calls


class TestSupportsToolCalls:
    """Tests for supports_tool_calls with unknown endpoints."""

    def test_unknown_endpoint_known_model(self, monkeypatch):
        """Unknown endpoint returns True when model is in the catalog."""
        monkeypatch.setenv("AI_API_ENDPOINT", "https://custom.api.example.com/v1")
        models = {"my-model": {"id": "my-model"}}
        assert supports_tool_calls("my-model", models) is True

    def test_unknown_endpoint_unknown_model(self, monkeypatch):
        """Unknown endpoint returns False when model is NOT in the catalog."""
        monkeypatch.setenv("AI_API_ENDPOINT", "https://custom.api.example.com/v1")
        models = {"other-model": {"id": "other-model"}}
        assert supports_tool_calls("missing-model", models) is False

    def test_copilot_endpoint_with_capabilities(self, monkeypatch):
        """Copilot endpoint checks capabilities.supports.tool_calls."""
        monkeypatch.setenv("AI_API_ENDPOINT", "https://api.githubcopilot.com")
        models = {
            "gpt-4o": {
                "id": "gpt-4o",
                "capabilities": {"supports": {"tool_calls": True}},
            }
        }
        assert supports_tool_calls("gpt-4o", models) is True

    def test_copilot_endpoint_without_capabilities(self, monkeypatch):
        """Copilot endpoint returns False when tool_calls not in capabilities."""
        monkeypatch.setenv("AI_API_ENDPOINT", "https://api.githubcopilot.com")
        models = {
            "text-only": {
                "id": "text-only",
                "capabilities": {"supports": {}},
            }
        }
        assert supports_tool_calls("text-only", models) is False

    def test_models_github_endpoint(self, monkeypatch):
        """models.github.ai checks for 'tool-calling' in capabilities list."""
        monkeypatch.setenv("AI_API_ENDPOINT", "https://models.github.ai/inference")
        models = {
            "openai/gpt-4o": {
                "id": "openai/gpt-4o",
                "capabilities": ["tool-calling", "chat"],
            }
        }
        assert supports_tool_calls("openai/gpt-4o", models) is True

    def test_models_github_endpoint_no_tool_calling(self, monkeypatch):
        """models.github.ai returns False when 'tool-calling' not in list."""
        monkeypatch.setenv("AI_API_ENDPOINT", "https://models.github.ai/inference")
        models = {
            "some-model": {
                "id": "some-model",
                "capabilities": ["chat"],
            }
        }
        assert supports_tool_calls("some-model", models) is False

    def test_openai_endpoint_gpt_model(self, monkeypatch):
        """OpenAI endpoint returns True for models containing 'gpt-'."""
        monkeypatch.setenv("AI_API_ENDPOINT", "https://api.openai.com/v1")
        assert supports_tool_calls("gpt-4o", {}) is True

    def test_openai_endpoint_non_gpt_model(self, monkeypatch):
        """OpenAI endpoint returns False for non-GPT models."""
        monkeypatch.setenv("AI_API_ENDPOINT", "https://api.openai.com/v1")
        assert supports_tool_calls("claude-3-opus", {}) is False


class TestAIAPIEndpointEnum:
    """Tests for the AI_API_ENDPOINT_ENUM StrEnum."""

    def test_enum_values(self):
        """All expected endpoint values exist."""
        assert AI_API_ENDPOINT_ENUM.AI_API_MODELS_GITHUB == "models.github.ai"
        assert AI_API_ENDPOINT_ENUM.AI_API_GITHUBCOPILOT == "api.githubcopilot.com"
        assert AI_API_ENDPOINT_ENUM.AI_API_OPENAI == "api.openai.com"

    def test_to_url_models_github(self):
        assert AI_API_ENDPOINT_ENUM.AI_API_MODELS_GITHUB.to_url() == "https://models.github.ai/inference"

    def test_to_url_copilot(self):
        assert AI_API_ENDPOINT_ENUM.AI_API_GITHUBCOPILOT.to_url() == "https://api.githubcopilot.com"

    def test_to_url_openai(self):
        assert AI_API_ENDPOINT_ENUM.AI_API_OPENAI.to_url() == "https://api.openai.com/v1"
