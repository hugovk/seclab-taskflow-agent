# SPDX-FileCopyrightText: 2025 GitHub
# SPDX-License-Identifier: MIT

import logging

# from mcp.server.fastmcp import FastMCP
from fastmcp import FastMCP  # move to FastMCP 2.0

from seclab_taskflow_agent.path_utils import log_file_name

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=log_file_name("mcp_echo.log"),
    filemode="a",
)

mcp = FastMCP("Echo")


@mcp.resource("echo://1/{message}")
def echo_resource1(message: str) -> str:
    """Echo a message as a resource"""
    return f"Resource 1 echo: {message}"


@mcp.resource("echo://2/{message}")
def echo_resource2(message: str) -> str:
    """Echo a message as a resource"""
    return f"Resource 2 echo: {message}"


@mcp.tool()
def echo_tool(message: str) -> str:
    """Echo a message as a tool"""
    return f"Tool echo: {message}"


@mcp.prompt()
def echo_prompt(message: str) -> str:
    """Create an echo prompt"""
    return f"Please process this message: {message}"


if __name__ == "__main__":
    mcp.run(show_banner=False)
