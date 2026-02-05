# SPDX-FileCopyrightText: 2025 GitHub
# SPDX-License-Identifier: MIT

"""
Test API endpoint configuration.
"""

import os
from urllib.parse import urlparse

import pytest

from seclab_taskflow_agent.capi import AI_API_ENDPOINT_ENUM, get_AI_endpoint, list_capi_models


class TestAPIEndpoint:
    """Test API endpoint configuration."""

    def test_default_api_endpoint(self):
        """Test that default API endpoint is set to models.github.ai/inference."""
        # When no env var is set, it should default to models.github.ai/inference
        try:
            # Save original env
            original_env = os.environ.pop("AI_API_ENDPOINT", None)
            endpoint = get_AI_endpoint()
            assert endpoint is not None
            assert isinstance(endpoint, str)
            assert urlparse(endpoint).netloc == AI_API_ENDPOINT_ENUM.AI_API_MODELS_GITHUB
        finally:
            # Restore original env
            if original_env:
                os.environ["AI_API_ENDPOINT"] = original_env

    def test_api_endpoint_env_override(self):
        """Test that AI_API_ENDPOINT can be overridden by environment variable."""
        try:
            # Save original env
            original_env = os.environ.pop("AI_API_ENDPOINT", None)
            # Set different endpoint
            test_endpoint = "https://api.githubcopilot.com"
            os.environ["AI_API_ENDPOINT"] = test_endpoint

            assert get_AI_endpoint() == test_endpoint
        finally:
            # Restore original env
            if original_env:
                os.environ["AI_API_ENDPOINT"] = original_env

    def test_to_url_models_github(self):
        """Test to_url method for models.github.ai endpoint."""
        endpoint = AI_API_ENDPOINT_ENUM.AI_API_MODELS_GITHUB
        assert endpoint.to_url() == "https://models.github.ai/inference"

    def test_to_url_githubcopilot(self):
        """Test to_url method for GitHub Copilot endpoint."""
        endpoint = AI_API_ENDPOINT_ENUM.AI_API_GITHUBCOPILOT
        assert endpoint.to_url() == "https://api.githubcopilot.com"

    def test_to_url_openai(self):
        """Test to_url method for OpenAI endpoint."""
        endpoint = AI_API_ENDPOINT_ENUM.AI_API_OPENAI
        assert endpoint.to_url() == "https://api.openai.com/v1"

    def test_unsupported_endpoint(self, monkeypatch):
        """Test that unsupported API endpoint raises ValueError."""
        api_endpoint = "https://unsupported.example.com"
        monkeypatch.setenv("AI_API_ENDPOINT", api_endpoint)
        with pytest.raises(ValueError) as excinfo:
            list_capi_models("abc")
        msg = str(excinfo.value)
        assert "Unsupported Model Endpoint" in msg
        assert "https://models.github.ai/inference" in msg
        assert "https://api.githubcopilot.com" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
