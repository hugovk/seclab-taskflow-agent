# Seclab Taskflow Agent

The Security Lab Taskflow Agent is an MCP enabled multi-Agent framework.

The Taskflow Agent is built on top of the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/).

While the Taskflow Agent does not integrate into the GitHub Doctom Copilot UX, it does operate using the Copilot API (CAPI) as its backend, similar to Copilot IDE extensions.

## Core Concepts

The Taskflow Agent leverages a GitHub Workflow-esque YAML based grammar to perform a series of tasks using a set of Agents.

Its primary value proposition is as a CLI tool that allows users to quickly define and script Agentic workflows without having to write any code.

Agents are defined through [personalities](personalities/), that receive a [task](taskflows/) to complete given a set of [tools](toolboxes/).

Agents can cooperate to complete sequences of tasks through so-called [taskflows](taskflows/GRAMMAR.md).

You can find a detailed overview of the taskflow grammar [here](https://github.com/GitHubSecurityLab/seclab-taskflow-agent/blob/main/taskflows/GRAMMAR.md) and example taskflows [here](https://github.com/GitHubSecurityLab/seclab-taskflow-agent/tree/main/taskflows/examples).

## Use Cases and Examples

The Seclab Taskflow Agent framework was primarily designed to fit the iterative feedback loop driven work involved in Agentic security research workflows and vulnerability triage tasks.

Its design philosophy is centered around the belief that a prompt level focus of capturing vulnerability patterns will greatly improve and scale security research results as frontier model capabilities evolve over time.

While the maintainer himself primarily uses this framework as a code auditing tool it also serves as a more generic swiss army knife for exploring Agentic workflows. For example, the GitHub Security Lab also uses this framework for automated code scanning alert triage.

The framework includes a [CodeQL](https://codeql.github.com/) MCP server that can be used for Agentic code review, see the [CVE-2023-2283](https://github.com/GitHubSecurityLab/seclab-taskflow-agent/blob/main/taskflows/CVE-2023-2283/CVE-2023-2283.yaml) for an example of how to have an Agent review C code using a CodeQL database ([demo video](https://www.youtube.com/watch?v=eRSPSVW8RMo)).

Instead of generating CodeQL queries itself, the CodeQL MCP Server is used to provide CodeQL-query based MCP tools that allow an Agent to navigate and explore code. It leverages templated CodeQL queries to provide targeted context for model driven code analysis.

## Requirements

Python >= 3.9 or Docker

## Configuration

Provide a GitHub token for an account that is entitled to use GitHub Copilot via the `COPILOT_TOKEN` environment variable. Further configuration is use case dependent, i.e. pending which MCP servers you'd like to use in your taskflows.

You can set persisting environment variables via an `.env` file in the project root.

Example:

```sh
# Tokens
COPILOT_TOKEN=<your_github_token>
# MCP configs
GITHUB_PERSONAL_ACCESS_TOKEN=<your_github_token>
CODEQL_DBS_BASE_PATH="/app/my_data/codeql_databases"
```

## Deploying from Source

First install the required dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Then run `python main.py`.

Example: deploying a prompt to an Agent Personality:

```sh
python main.py -p assistant 'explain modems to me please'
```

Example: deploying a Taskflow:

```sh
python main.py -t example
```

## Deploying from Docker

You can deploy the Taskflow Agent via its Docker image using `docker/run.sh`.

WARNING: the Agent Docker image is _NOT_ intended as a security boundary but strictly a deployment convenience.

The image entrypoint is `main.py` and thus it operates the same as invoking the Agent from source directly.

You can find the Docker image for the Seclab Taskflow Agent [here](https://github.com/GitHubSecurityLab/seclab-taskflow-agent/pkgs/container/seclab-taskflow-agent) and how it is built [here](release_tools/).

Note that this image is based on a public release of the Taskflow Agent, and you will have to mount any custom taskflows, personalities, or prompts into the image for them to be available to the Agent.

Optional image mount points to supply custom data are configured via the environment:

- Custom data via `MY_DATA`, mounts to `/app/my_data`
- Custom personalities via `MY_PERSONALITIES`, mounts to `/app/personalities/my_personalities`
- Custom taskflows via `MY_TASKFLOWS`, mounts to `/app/taskflows/my_taskflows`
- Custom prompts via `MY_PROMPTS`, mounts to `/app/prompts/my_prompts`
- Custom toolboxes via `MY_TOOLBOXES`, mounts to `/app/toolboxes/my_toolboxes`

See [docker/run.sh](docker/run.sh) for further details.

Example: deploying a Taskflow (example.yaml):

```sh
docker/run.sh -t example
```
Example: deploying a custom taskflow (custom_taskflow.yaml):

```sh
MY_TASKFLOWS=~/my_taskflows docker/run.sh -t custom_taskflow
```

Example: deploying a custom taskflow (custom_taskflow.yaml) and making local CodeQL databases available to the CodeQL MCP server:

```sh
MY_TASKFLOWS=~/my_taskflows MY_DATA=~/app/my_data CODEQL_DBS_BASE_PATH=/app/my_data/codeql_databases docker/run.sh -t custom_taskflow
```

For more advanced scenarios like e.g. making custom MCP server code available, you can alter the run script to mount your custom code into the image and configure your toolboxes to use said code accordingly.

```sh
export MY_MCP_SERVERS="$PWD"/mcp_servers
export MY_TOOLBOXES="$PWD"/toolboxes
export MY_PERSONALITIES="$PWD"/personalities
export MY_TASKFLOWS="$PWD"/taskflows
export MY_PROMPTS="$PWD"/prompts
export MY_DATA="$PWD"/data

if [ ! -f ".env" ]; then
    touch ".env"
fi

docker run \
       --volume "$PWD"/logs:/app/logs \
       --mount type=bind,src="$PWD"/.env,dst=/app/.env,ro \
       ${MY_DATA:+--mount type=bind,src=$MY_DATA,dst=/app/my_data} \
       ${MY_MCP_SERVERS:+--mount type=bind,src=$MY_MCP_SERVERS,dst=/app/my_mcp_servers,ro} \
       ${MY_TASKFLOWS:+--mount type=bind,src=$MY_TASKFLOWS,dst=/app/taskflows/my_taskflows,ro} \
       ${MY_TOOLBOXES:+--mount type=bind,src=$MY_TOOLBOXES,dst=/app/toolboxes/my_toolboxes,ro} \
       ${MY_PROMPTS:+--mount type=bind,src=$MY_PROMPTS,dst=/app/prompts/my_prompts,ro} \
       ${MY_PERSONALITIES:+--mount type=bind,src=$MY_PERSONALITIES,dst=/app/personalities/my_personalities,ro} \
       "ghcr.io/githubsecuritylab/seclab-taskflow-agent" "$@"
```

## General YAML file headers

Every YAML files used by the Seclab Taskflow Agent must include a header like this:

```yaml
seclab-taskflow-agent:
  version: 1
  filetype: taskflow
  filekey: GitHubSecurityLab/seclab-taskflow-agent/taskflows/CVE-2023-2283/CVE-2023-2283
```

The `filetype` determines whether the file defines a personality, toolbox, or
taskflow. This means that different types of files can be stored in the same directory.
A `filetype` can be one of the followings:
  - taskflow
  - personality
  - toolbox
  - prompt
  - model_config

We'll explain these file types in more detail in the following sections.

The `filekey` is a unique name for the file. It is used to allow
cross-referencing between files. For example, a taskflow can reference
a personality by its filekey. Because filekeys are used for
cross-referencing (rather than file paths), it means that you can move
a file to a different directory without breaking the links. This also
means that you can easily import new files by dropping them into a sub-directory.
We recommend including something like your
GitHub `<username>/<reponame>` in your filekeys to make them globally unique.

In the above example, it is a `taskflow` file with `filekey` `GitHubSecurityLab/seclab-taskflow-agent/taskflows/CVE-2023-2283/CVE-2023-2283`. The `filekey` is needed to run the taskflow from command line, e.g.:

```
python3 main.py -t GitHubSecurityLab/seclab-taskflow-agent/taskflows/CVE-2023-2283/CVE-2023-2283
```

will run the taskflow.

The `version` number in the header should always be 1. It means that the
file uses version 1 of the seclab-taskflow-agent syntax. If we ever need
to make a major change to the syntax, then we'll update the version number.
This will hopefully enable us to make changes without breaking backwards
compatibility.

We'll now explain the role of different types of files and functionalities available to them.

## Personalities

Core characteristics for a single Agent. Configured through YAML files of `filetype` `personality`.

These are system prompt level instructions.

Example:

```yaml
# personalities define the system prompt level directives for this Agent
seclab-taskflow-agent:
  version: 1
  filetype: personality
  filekey: GitHubSecurityLab/seclab-taskflow-agent/personalities/examples/echo

personality: |
  You are a simple echo bot. You use echo tools to echo things.

task: |
  Echo user inputs using the echo tools.

# personality toolboxes map to mcp servers made available to this Agent
toolboxes:
  - GitHubSecurityLab/seclab-taskflow-agent/toolboxes/echo
```

In the above, the `personality` and `task` field specifies the system prompt to be used whenever this `personality` is used.
The `toolboxes` are the tools that are available to this `personality`. The `toolboxes` should be a list of `filekey` specifying 
files of the `filetype` `toolbox`. 

Personalities can be used in two ways. First it can be used standalone with a prompt input from the command line:

```
python3 main.py -p GitHubSecurityLab/seclab-taskflow-agent/personalities/examples/echo "echo this message"
```

In this case, `personality` and `task` from `GitHubSecurityLab/seclab-taskflow-agent/personalities/examples/echo` are used as the 
system prompt while the user argument `echo this message` is used as a user prompt. In this use case, the only tools that this 
personality has access to is the `toolboxes` specified in the file.

Personalities can also be used in a `taskflow` to perform tasks. This is done by adding the `personality` to the `agents` field in a `taskflow` file:

```yaml
taskflow:
  - task:
      ...
      agents:
        - GitHubSecurityLab/seclab-taskflow-agent/personalities/assistant
      user_prompt: |
        Fetch all the open pull requests from `github/codeql` github repository. 
        You do not need to provide a summary of the results.
      toolboxes:
        - GitHubSecurityLab/seclab-taskflow-agent/toolboxes/github_official
```

In this case, the `personality` specified in `agents` provides the system prompt and the user prompt is specified in `user_prompt` field of the task. A big difference in this case is that the `toolboxes` specified in the `task` will overwrite the `toolboxes` that the `personality` has access to. So in the above example, the `GitHubSecurityLab/seclab-taskflow-agent/personalities/assistant` will have access to the `GitHubSecurityLab/seclab-taskflow-agent/toolboxes/github_official` toolbox instead of its own toolbox. It is important to note that the `personalities` toolboxes get *overwritten* in this case, so whenever a `toolboxes` field is provided in a `task`, it'll use the provided toolboxes and `personality` loses access to its own toolboxes. e.g.

```yaml
taskflow:
  - task:
      ...
      agents:
        - GitHubSecurityLab/seclab-taskflow-agent/personalities/examples/echo
      user_prompt: |
        echo this
      toolboxes:
        - GitHubSecurityLab/seclab-taskflow-agent/toolboxes/github_official
```

In the above `task`, `GitHubSecurityLab/seclab-taskflow-agent/personalities/examples/echo` will only have access to the `GitHubSecurityLab/seclab-taskflow-agent/toolboxes/github_official` and can no longer access the `GitHubSecurityLab/seclab-taskflow-agent/toolboxes/echo` `toolbox`. (Unless it is added also in the `task` `toolboxes`)

## Toolboxes

MCP servers that provide tools. Configured through YAML files of `filetype` `toolboxes`. These are files that provide
the type and parameters to start an MCP server. 

For example, to start a stdio MCP server that are implemented in a python file:

```yaml
# stdio mcp server configuration
seclab-taskflow-agent:
  version: 1
  filetype: toolbox
  filekey: toolboxes/echo

server_params:
  kind: stdio
  command: python
  args:
    - toolboxes/echo/echo.py
  env:
    SOME: value
```

In the above, `command` and `args` are just the command and arguments that should be run to start the MCP server. Environment variables can be passed using the `env` field.

A [streamable](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#streamable-http) is also supported by specifying the `kind` to `streamable`:

```yaml
server_params:
  kind: streamable
  url: https://api.githubcopilot.com/mcp/
  #See https://github.com/github/github-mcp-server/blob/main/docs/remote-server.md
  headers:
    Authorization: "{{ env GITHUB_AUTH_HEADER }}"
  optional_headers:
    X-MCP-Toolsets: "{{ env GITHUB_MCP_TOOLSETS }}"
    X-MCP-Readonly: "{{ env GITHUB_MCP_READONLY }}"
```

You can force certain tools within a `toolbox` to require user confirmation to run. This can be helpful if a tool may perform irreversible actions and should require user approval prior to its use. This is done by including the name of the tool (function) in the MCP server in the `confirm` section:

```yaml
server_params:
  kind: stdio
  ...
# the list of tools that you want the framework to confirm with the user before executing
# use this to guard rail any potentially dangerous functions from MCP servers
confirm:
  - memcache_clear_cache
```

## Taskflows

A sequence of interdependent tasks performed by a set of Agents. Configured through YAML files of `filetype` `taskflow`. 
Taskflows supports a number of features, and their details can be found [here](taskflows/GRAMMAR.md).

Example:

```yaml
seclab-taskflow-agent:
  version: 1
  filetype: taskflow
  filekey: taskflows/examples/example.yaml

taskflow:
  - task:
      # taskflows can optionally choose any of the support CAPI models for a task
      model: gpt-4.1
      # taskflows can optionally limit the max allowed number of Agent task loop
      # iterations to complete a task, this defaults to 50 when not provided
      max_steps: 20
      must_complete: true
      # taskflows can set a primary (first entry) and handoff (additional entries) agent
      agents:
        - personalities/c_auditer.yaml
        - personalities/examples/fruit_expert.yaml
      user_prompt: |
        Store an example vulnerable C program that uses `strcpy` in the
        `vulnerable_c_example` memory key and explain why `strcpy`
        is insecure in the C programming language. Do this before handing off
        to any other agent.

        Finally, why are apples and oranges healthy to eat?

      # taskflows can set temporary environment variables, these support the general
      # "{{ env FROM_EXISTING_ENVIRONMENT }" pattern we use elsewhere as well
      # these environment variables can then be made available to any stdio mcp server
      # through its respective yaml configuration, see memcache.yaml for an example
      # you can use these to override top-level environment variables on a per-task basis
      env:
        MEMCACHE_STATE_DIR: "example_taskflow/"
        MEMCACHE_BACKEND: "dictionary_file"
      # taskflows can optionally override personality toolboxes, in this example
      # this normally only has the memcache toolbox, but we extend it here with
      # the GHSA toolbox
      toolboxes:
        - toolboxes/memcache.yaml
        - toolboxes/codeql.yaml
  - task:
      must_complete: true
      model: gpt-4.1
      agents:
        - personalities/c_auditer.yaml
      user_prompt: |
        Retrieve C code for security review from the `vulnerable_c_example`
        memory key and perform a review.

        Clear the memory cache when you're done.
      env:
        MEMCACHE_STATE_DIR: "example_taskflow/"
        MEMCACHE_BACKEND: "dictionary_file"
      toolboxes:
        - toolboxes/memcache.yaml
      # headless mode does not prompt for tool call confirms configured for a server
      # note: this will auto-allow, if you want control over potentially dangerous
      # tool calls, then you should NOT run a task in headless mode (default: false)
      headless: true
  - task:
      # tasks can also run shell scripts that return e.g. json output for repeat prompt iterable
      must_complete: true
      run: |
        echo '["apple", "banana", "orange"]'
  - task:
      repeat_prompt: true
      agents:
        - personalities/assistant.yaml
      user_prompt: |
        What kind of fruit is {{ RESULT }}?
```

Taskflows support [Agent handoffs](https://openai.github.io/openai-agents-python/handoffs/). Handoffs are useful for implementing triage patterns where the primary Agent can decide to handoff a task to any subsequent Agents in the `Agents` list.

See the [taskflow examples](taskflows/examples) for other useful Taskflow patterns such as repeatable and asynchronous templated prompts.

## Prompt

Prompts are configured through YAML files of `filetype` `prompt`. They define a reusable prompt that can be referenced in `taskflow` files.

They contain only one field, the `prompt` field, which is used to replace any `{{ PROMPT_<filekey> }}` template parameter in a taskflow. For example, the following `prompt`.

```yaml
seclab-taskflow-agent:
  version: 1
  filetype: prompt
  filekey: GitHubSecurityLab/seclab-taskflow-agent/prompts/examples/example_prompt

prompt: |
  Tell me more about bananas as well.
```

would replace any `{{ PROMPT_GitHubSecurityLab/seclab-taskflow-agent/prompts/examples/example_prompt }}` template parameter found in the `user_prompt` section in a taskflow:

```yaml
  - task:
      agents:
        - fruit_expert
      user_prompt: |
        Tell me more about apples.

        {{ PROMPTS_GitHubSecurityLab/seclab-taskflow-agent/prompts/examples/example_prompt }}
```

becomes:

```yaml
  - task:
      agents:
        - fruit_expert
      user_prompt: |
        Tell me more about apples.

        Tell me more about bananas as well.
```

## Model configs

Model configs are configured through YAML files of `filetype` `model_config`. These provide a way to configure model versions.

```yaml
seclab-taskflow-agent:
  version: 1
  filetype: model_config
  filekey: GitHubSecurityLab/seclab-taskflow-agent/configs/model_config
models:
   gpt_latest: gpt-5
```

A `model_config` file can be used in a `taskflow` and the values defined in `models` can then be used throughout.

```yaml
model_config: GitHubSecurityLab/seclab-taskflow-agent/configs/model_config

taskflow:
  - task:
      model: gpt_latest
```

Model version can then be updated by changing `gpt_latest` in the `model_config` file and applied across all taskflows that use the config.

## Passing environment variables

Files of types `taskflow` and `toolbox` allow environment variables to be passed using the `env` field:

```yaml
server_params:
  ...
  env:
    CODEQL_DBS_BASE_PATH: "{{ env CODEQL_DBS_BASE_PATH }}"
    # prevent git repo operations on gh codeql executions
    GH_NO_UPDATE_NOTIFIER: "Disable"
```

For `toolbox`, `env` can be used inside `server_params`. A template of the form `{{ env ENV_VARIABLE_NAME}}` can be used to pass values of the environment variable from the current process to the MCP server. So in the above, the MCP server is run with `GH_NO_UPDATE_NOTIFIER=disable` and passes the value of `CODEQL_DBS_BASE_PATH` from the current process to the MCP server. The templated paramater `{{ env CODEQL_DBS_BASE_PATH}}` is replaced by the value of the environment variable `CODEQL_DBS_BASE_PATH` in the current process. 

Similarly, environment variables can be passed to a `task` in a `taskflow`:

```yaml
taskflow:
  - task:
      must_complete: true
      agents:
        - GitHubSecurityLab/seclab-taskflow-agent/personalities/assistant
      user_prompt: |
        Store the json array ["apples", "oranges", "bananas"] in the `fruits` memory key.
      env:
        MEMCACHE_STATE_DIR: "example_taskflow/"
        MEMCACHE_BACKEND: "dictionary_file"
```

This overwrites the environment variables `MEMCACHE_STATE_DIR` and `MEMCACHE_BACKEND` for the task only. A template `{{ env ENV_VARIABLE_NAME}}` can also be used.

Note that when using the template `{{ env ENV_VARIABLE_NAME }}`, `ENV_VARIABLE_NAME` must be the name of an environment variable in the current process.

## License

This project is licensed under the terms of the MIT open source license. Please refer to the [LICENSE](./LICENSE) file for the full terms.

## Maintainers

[CODEOWNERS](./CODEOWNERS)

## Support

[SUPPORT](./SUPPORT.md)

## Acknowledgements

Security Lab team members [Man Yue Mo](https://github.com/m-y-mo) and [Peter Stockli](https://github.com/p-) for contributing heavily to the testing and development of this framework, as well as the rest of the Security Lab team for helpful discussions and feedback.
