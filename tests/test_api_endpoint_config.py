# SPDX-FileCopyrightText: 2025 GitHub
# SPDX-License-Identifier: MIT

"""
Test API endpoint configuration.
"""

import os
from urllib.parse import urlparse

import pytest

from seclab_taskflow_agent.capi import AI_API_ENDPOINT_ENUM, get_AI_endpoint


class TestAPIEndpoint:
    """Test API endpoint configuration."""

    def test_default_api_endpoint(self):
        """Test that default API endpoint is set to models.github.ai/inference."""
        # When no env var is set, it should default to models.github.ai/inference
        try:
            # Save original env
            original_env = os.environ.pop('AI_API_ENDPOINT', None)
            endpoint = get_AI_endpoint()
            assert endpoint is not None
            assert isinstance(endpoint, str)
            assert urlparse(endpoint).netloc == AI_API_ENDPOINT_ENUM.AI_API_MODELS_GITHUB
        finally:
            # Restore original env
            if original_env:
                os.environ['AI_API_ENDPOINT'] = original_env

    def test_api_endpoint_env_override(self):
        """Test that AI_API_ENDPOINT can be overridden by environment variable."""
        try:
            # Save original env
            original_env = os.environ.pop('AI_API_ENDPOINT', None)
            # Set different endpoint
            test_endpoint = 'https://api.githubcopilot.com'
            os.environ['AI_API_ENDPOINT'] = test_endpoint

            assert get_AI_endpoint() == test_endpoint
        finally:
            # Restore original env
            if original_env:
                os.environ['AI_API_ENDPOINT'] = original_env

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
