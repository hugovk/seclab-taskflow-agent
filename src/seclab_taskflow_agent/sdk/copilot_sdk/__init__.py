# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""GitHub Copilot SDK backend adapter."""

from __future__ import annotations

__all__ = ["CopilotSDKBackend"]

from .. import register_backend
from .backend import CopilotSDKBackend

register_backend(CopilotSDKBackend())
