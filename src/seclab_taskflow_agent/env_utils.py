# SPDX-FileCopyrightText: 2025 GitHub
# SPDX-License-Identifier: MIT

import os
import re


def swap_env(s):
    match = re.search(r"{{\s*(env)\s+([A-Z0-9_]+)\s*}}", s)
    if match and not os.getenv(match.group(2)):
        raise LookupError(f"Requested {match.group(2)} from env but it does not exist!")
    return os.getenv(match.group(2)) if match else s


class TmpEnv:
    def __init__(self, env):
        self.env = dict(env)
        self.restore_env = dict(os.environ)

    def __enter__(self):
        for k, v in self.env.items():
            os.environ[k] = swap_env(v)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for k, v in self.env.items():
            del os.environ[k]
            if k in self.restore_env:
                os.environ[k] = self.restore_env[k]
