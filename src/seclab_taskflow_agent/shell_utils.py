# SPDX-FileCopyrightText: 2025 GitHub
# SPDX-License-Identifier: MIT

import subprocess
import tempfile
import logging

from mcp.types import CallToolResult, TextContent


def shell_command_to_string(cmd):
    logging.info(f"Executing: {cmd}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    stdout, stderr = p.communicate()
    p.wait()
    if p.returncode:
        raise RuntimeError(stderr)
    return stdout


def shell_exec_with_temporary_file(script, shell="bash"):
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_file:
        temp_file.write(script)
        temp_file.flush()
        result = shell_command_to_string([shell, temp_file.name])
        return result


def shell_tool_call(run):
    stdout = shell_exec_with_temporary_file(run)
    # this allows e.g. shell based jq output to become available for repeat prompts
    result = CallToolResult(content=[TextContent(type="text", text=stdout, annotations=None, meta=None)])
    return result
