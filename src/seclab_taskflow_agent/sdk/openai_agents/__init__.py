# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""openai-agents backend adapter package.

Importing this module registers :class:`OpenAIAgentsBackend` with the
SDK registry so ``sdk.get_backend("openai_agents")`` returns a usable
instance.
"""

from __future__ import annotations

__all__ = ["OpenAIAgentsBackend"]

from .. import register_backend
from .backend import OpenAIAgentsBackend

register_backend(OpenAIAgentsBackend())
