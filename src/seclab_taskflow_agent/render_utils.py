# SPDX-FileCopyrightText: 2025 GitHub
# SPDX-License-Identifier: MIT

import asyncio
import logging

from .path_utils import log_file_name

async_output = {}
async_output_lock = asyncio.Lock()

render_logger = logging.getLogger("render")
file_handler = logging.FileHandler(log_file_name("render_stdout.log"))
file_handler.terminator = ""
render_logger.addHandler(file_handler)
render_logger.propagate = False


async def flush_async_output(task_id: str):
    async with async_output_lock:
        if task_id not in async_output:
            raise ValueError(f"No async output for task: {task_id}")
        data = async_output[task_id]
        del async_output[task_id]
    await render_model_output(f"** 🤖✏️ Output for async task: {task_id}\n\n")
    await render_model_output(data)


async def render_model_output(data: str, log: bool = True, async_task: bool = False, task_id: str | None = None):
    async with async_output_lock:
        if async_task and task_id:
            if task_id in async_output:
                async_output[task_id] += data
                data = ""
            else:
                async_output[task_id] = data
                data = "** 🤖✏️ Gathering output from async task ... please hold\n"
    if data:
        if log:
            render_logger.info(data)
        print(data, end="", flush=True)
