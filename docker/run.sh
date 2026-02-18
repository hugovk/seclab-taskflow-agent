#!/bin/bash

# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

# Script for running seclab-taskflow-agent in a docker container
# (using the docker image that we publish).
#
# To use this script, `cd` to a directory containing taskflows.
# For example:
#
#   git clone https://github.com/GitHubSecurityLab/seclab-taskflow-agent.git
#   cd seclab-taskflow-agent/src
#   export AI_API_TOKEN=<My GitHub PAT>
#   export GH_TOKEN=<My GitHub PAT>
#   sudo -E ../docker/run.sh -p seclab_taskflow_agent.personalities.assistant 'explain modems to me please'

touch -a .env

docker run -i \
       --mount type=bind,src="$PWD",dst=/app \
       -e GH_TOKEN="$GH_TOKEN" \
       -e AI_API_TOKEN="$AI_API_TOKEN" \
       "ghcr.io/githubsecuritylab/seclab-taskflow-agent" "$@"
