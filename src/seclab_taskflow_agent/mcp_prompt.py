# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""MCP system-prompt construction.

Assembles the full system prompt from a base personality prompt, available
tools/resources, important guidelines, and server-supplied instructions.
"""

from __future__ import annotations

__all__ = ["mcp_system_prompt"]


def mcp_system_prompt(
    system_prompt: str,
    task: str,
    tools: list[str] | None = None,
    resources: list[str] | None = None,
    resource_templates: list[str] | None = None,
    important_guidelines: list[str] | None = None,
    server_prompts: list[str] | None = None,
) -> str:
    """Build a well-structured system prompt for an MCP agent.

    Each optional section is appended only when its list is non-empty.

    Args:
        system_prompt: Base personality / instruction text.
        task: The primary task description for the agent.
        tools: Human-readable tool descriptions.
        resources: Human-readable resource descriptions.
        resource_templates: Human-readable resource-template descriptions.
        important_guidelines: Critical behavioural constraints.
        server_prompts: Additional guidance supplied by MCP servers.

    Returns:
        The fully assembled system prompt string.
    """
    if tools is None:
        tools = []
    if resources is None:
        resources = []
    if resource_templates is None:
        resource_templates = []
    if important_guidelines is None:
        important_guidelines = []
    if server_prompts is None:
        server_prompts = []

    prompt = f"""
{system_prompt}
"""

    if tools:
        prompt += """

# Available Tools

- {tools}
""".format(tools="\n- ".join(tools))

    if resources:
        prompt += """

# Available Resources

- {resources}
""".format(resources="\n- ".join(resources))

    if resource_templates:
        prompt += """

# Available Resource Templates

- {resource_templates}
""".format(resource_templates="\n- ".join(resource_templates))

    if important_guidelines:
        prompt += """

# Important Guidelines

- IMPORTANT: {guidelines}
""".format(guidelines="\n- IMPORTANT: ".join(important_guidelines))

    if server_prompts:
        prompt += """

# Additional Guidelines

{server_prompts}

""".format(server_prompts="\n\n".join(server_prompts))

    if task:
        prompt += f"""

# Primary Task to Complete

{task}

"""

    return prompt
