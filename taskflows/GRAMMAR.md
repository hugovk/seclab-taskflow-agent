# Taskflows

Taskflows are YAML lists of tasks. They are specified by the `filetype` `taskflow`.

Example:

```yaml
taskflow:
  - task:
    ...
  - task:
    ...
```

## Tasks

Tasks define, at minimum, a list of Agents to use and a User Prompt.

Example:

```yaml
  - task:
      agents:
        - assistant
      user_prompt: |
        This is a user prompt.
```

Note: The exception to this rule are `run` shell tasks.

### Agents

Agents define the system prompt to be used for the task. It contains a list of `filekey` pointing to files of `personality` `filetype`.

For example, to use the `personality` defined in the following:

```yaml
seclab-taskflow-agent:
  version: 1
  filetype: personality
  filekey: GitHubSecurityLab/seclab-taskflow-agent/personalities/assistant

personality: |
  You are a helpful assistant.
  
task: |
  Your primary task is to use available tools to complete user defined tasks.

  Always use available tools to complete your tasks. If the tools you require
  to complete a task are not available, politely decline the task.
  
toolboxes:
  - ...
```

The task should include the `filekey` in its list of `agents`:

```yaml
  - task:
      agents:
        - GitHubSecurityLab/seclab-taskflow-agent/personalities/assistant
  ...
```

Task agent lists can define one (primary) or more (handoff) agents.

Example:

```yaml
  - task:
      agents:
        - primary_agent
        - handoff_agent1
        - ...
        - handoff_agentN
      user_prompt: |
        ...
```

### Model

Tasks can optionally specify which Model to use on the configured inference endpoint:

```yaml
  - task:
      model: gpt-4.1
      agents:
        - assistant
      user_prompt: |
        This is a user prompt.
```

Note that model identifiers may differ between OpenAI compatible endpoint providers, make sure you change your model identifier accordingly when switching providers. If not specified, a default LLM model (`gpt-4o`) is used.

### Completion Requirement

Tasks can be marked as requiring completion, if a required task fails, the taskflow will abort. This defaults to false.

Example:

```yaml
  - task:
      must_complete: true
      agents:
        - assistant
      user_prompt: |
        ...
```

### Running templated tasks in a loop

Often we may want to iterate through the same tasks with different inputs. For example, we may want to do fetch all the functions from a code base and then analyze each of the function. This can be done using two consecutive task and with the help of the `repeat_prompt` field. 

```yaml
  - task:
    agents:
      - assistant
    user_prompt: |
      Fetch all the functions in the code base and create a list with entries of the form {'name' : <function_name>, 'body' : <function_body>}
  - task:
    repeat_prompt: true
    agents:
      - c_auditer
    user_prompt: |
      The function has name {{ RESULT_name }} and body {{ RESULT_body }} analyze the function.
```

In the above, the first task fetches functions in the code base and create a json list object, with each entry having a `name` and `body` field. In the next task, `repeat_prompt` is set to true, meaning that a task is created for each individual object in the list and the object fields are referenced in the templated prompt using `{{ RESULT_<fieldname> }}`. In other words, `{{ RESULT_name }}` in the prompt is replaced with the value of the `name` field of the object etc. For example, if the list of functions fetched from the first task is:

```javascript
[{'name' : foo, 'body' : foo(){return 1;}}, {'name' : bar, 'body' : bar(a) {return a + 1;}}]
```

Then the tasks created will have their prompts replaced by:

```yaml
      The function has name foo and body foo(){return 1;} analyze the function.
```

etc. 

Note that when using `repeat_prompt`, the last tool call result of the previous task is used as the iterable. It is recommended to keep the task that creates the iterable short and simple (e.g. just make one tool call to fetch a list of results) to avoid wrong results being passed to the repeat prompt.

The iterable can also contain a list of primitives like string or number, in which case, the template `{{ RESULT }}` can be used in the `repeat_prompt` prompt to parse the results instead:

```yaml
  - task:
      max_steps: 5
      must_complete: true
      agents:
        - GitHubSecurityLab/seclab-taskflow-agent/personalities/assistant
      user_prompt: |
        Store the json array [1, 2, 3] in memory under the
        `test_repeat_prompt` key as a json object, then retrieve
        the contents of the `test_repeat_prompt` key from memory
        ...
  - task:
      # if the last mcp tool result is iterable
      # repeat_prompt can iter those results
      must_complete: true
      repeat_prompt: true
      agents:
        - GitHubSecurityLab/seclab-taskflow-agent/personalities/assistant
      user_prompt: |
        What is the integer value of {{ RESULT }}?
```

Repeat prompt can be run in parallel by setting the `async` field to `true`:

```yaml
  - task:
    repeat_prompt: true
    async: true
    agents:
      - c_auditer
    user_prompt: |
      The function has name {{ RESULT_name }} and body {{ RESULT_body }} analyze the function.
```

An optional limit can be set to limit the number of asynchronous tasks via `async_limit`. If not set, the default value (5) is used.

```yaml
  - task:
    repeat_prompt: true
    async: true
    async_limit: 3
    agents:
      - c_auditer
    user_prompt: |
      The function has name {{ RESULT_name }} and body {{ RESULT_body }} analyze the function.
```

Both `async` and `async_limit` have no effect when use outside of a `repeat_prompt`.

At the moment, we do not support nested `repeat_prompt`. So the following is not allowed:

```yaml
  - task:
    repeat_prompt: true
    agents:
      - c_auditer
    user_prompt: |
      The function has name {{ RESULT_name }} and body {{ RESULT_body }} analyze the function.
  - task:
    repeat_prompt: true
    ...
```

#### Shell Tasks

Tasks can be entirely shell based through the run directive. This simply runs a shell command and pass the result directly to the next task. It is used for creating iterable results for `repeat_prompt`.

For example:

```yaml
  - task:
      must_complete: true
      run: |
        echo '["apple", "banana", "orange"]'
  - task:
      repeat_prompt: true
      agents:
        - assistant
      user_prompt: |
        What kind of fruit is {{ RESULT }}?
```

The string `["apple", "banana", "orange"]` is then passed directly to the next task.

This allows you to e.g. pass in json iterable outputs from shellscripts into a prompt task.

Use shell tasks when you want to iterate on results that don't need to be generated via a tool call.

#### Context Exclusion

Often when creating iterable results for a `repeat_prompt`, a large iterable is created and we do not want it to be passed to the LLM model because it can easily exceed the token limit. In this case, tasks can specify that their tool results and output should be available at the Agent level but not included in the Model context using the `exclude_from_context` field.

Example:

```yaml
  - task:
      exclude_from_context: true
      agents:
        - assistant
      user_prompt: |
        List all the files in the codeql database `some/codeql/db`.
      toolboxes:
        - codeql
```

### Toolboxes / MCP Servers

Toolboxes are MCP server configurations. They can be defined at the Agent level or overridden at the task level. These MCP servers are started and made available to the Agents in the Agents list during a Task. The `toolboxes` field should contain a list of `filekey` for the `toolboxes` that are available for the task:

```yaml
  - task:
      ...
      toolboxes:
        - GitHubSecurityLab/seclab-taskflow-agent/toolboxes/codeql
```

If no `toolboxes` is specified, then the `toolboxes` defined in the `personality` of the `agent` is used:

```yaml
   - task:
      agents:
        - GitHubSecurityLab/seclab-taskflow-agent/personalities/c_auditer
      user_prompt: |
        List all the files in the codeql database `some/codeql/db`.      
   - task:
```

In the above `task`, as no `toolboxes` is specified, the `toolboxes` defined in the `personality` of `GitHubSecurityLab/seclab-taskflow-agent/personalities/c_auditer` is used.

Note that when `toolboxes` is defined for a task, it *overwrites* the `toolboxes` that are available. For example, in the following `task`:

```yaml
   - task:
      agents:
        - GitHubSecurityLab/seclab-taskflow-agent/personalities/c_auditer
      user_prompt: |
        List all the files in the codeql database `some/codeql/db`.      
      toolboxes:
        - GitHubSecurityLab/seclab-taskflow-agent/toolboxes/echo

```

For this task, the `agent` `GitHubSecurityLab/seclab-taskflow-agent/personalities/c_auditer` will have access to the `GitHubSecurityLab/seclab-taskflow-agent/toolboxes/echo` tool.

### Headless Runs

MCP server configurations can request confirmations for tool calls. These confirmations are prompted on the terminal. If you want to allow all tool calls by default for headless use, you can set a task to run headless.

Example:

```yaml
  - task:
      headless: true
      agents:
        - assistant
      user_prompt: |
        Clear the memory cache.
      toolboxes:
        - memcache
```

### Environment Variables

Tasks can be configured to set temporary os environment variables available during the task. This is primarily used to pass through configuration options to toolboxes (mcp servers).

Example:

```yaml
  - task:
      headless: true
      agents:
        - assistant
      user_prompt: |
        Store `hello` in the memory key `world`.
      toolboxes:
        - memcache
      env:
        MEMCACHE_STATE_DIR: "example_taskflow/"
        MEMCACHE_BACKEND: "dictionary_file"
```

### Globals

Taskflows can define toplevel global variables available to every task.

Example:

```yaml
globals:
  fruit: bananas
taskflow:
  - task:
      agents:
        - fruit_expert
      user_prompt: |
        Tell me more about {{ GLOBALS_fruit }}.
```

### Reusable Tasks

Tasks can reuse single step taskflows and optionally override any of its configurations. This is done by setting a `uses` field with the `filekey` of the single step taskflow as its value.

Example:

```yaml
  - task:
      uses: single_step_taskflow
      model: gpt-4o
```

In this case, the prompt and settings of `single_step_taskflow` is used. However, the `model` parameter is overwritten by `gpt-4o`. For example, if `single_step_taskflow` looks like this:

```yaml
taskflow:
  - task:
      agents:
        - some_agent
      model:
        gpt-4.1
      user_prompt: |
        some actions
      toolboxes:
        - some_toolboxes
```

Then the `task` that uses it effectively becomes:
```yaml
  - task:
      agents:
        - some_agent
      model:
        gpt-4o
      user_prompt: |
        some actions
      toolboxes:
        - some_toolboxes
```

which all settings inherited from `single_step_taskflow` while `model` is overwritten.

Any `taskflow` that contains only a single step can be used as a reusable taskflow.

A reusable taskflow can also have templated prompt that takes inputs from its user. This is specified with the `inputs` field from the user.

```yaml
  - task:
      uses: single_step_taskflow
      inputs:
        fruit: apples
```

```yaml
  - task:
      agents:
        - fruit_expert
      user_prompt: |
        Tell me more about {{ INPUTS_fruit }}.
```

In this case, the template parameter `{{ INPUTS_fruit }}` is replaced by the value of `fruit` from the `inputs` of the user, which is apples in this case:

```yaml
  - task:
      agents:
        - fruit_expert
      user_prompt: |
        Tell me more about apples.
```

### Reusable Prompts

Reusable prompts are defined in files of `filetype` `prompts`. These are like macros that gets replaced when a templated parameter of the form `{{ PROMPTS_<filekey> }}` is encountered.

Tasks can incorporate templated prompts which are then replaced by the actual prompt. For example:

Example:

```yaml
  - task:
      agents:
        - fruit_expert
      user_prompt: |
        Tell me more about apples.

        {{ PROMPTS_GitHubSecurityLab/seclab-taskflow-agent/prompts/examples/example_prompt }}
```
and `GitHubSecurityLab/seclab-taskflow-agent/prompts/examples/example_prompt` is the following:

```yaml
seclab-taskflow-agent:
  version: 1
  filetype: prompt
  filekey: GitHubSecurityLab/seclab-taskflow-agent/prompts/examples/example_prompt

prompt: |
  Tell me more about bananas as well.
```

Then the actual task becomes:

```yaml
  - task:
      agents:
        - fruit_expert
      user_prompt: |
        Tell me more about apples.

        Tell me more about bananas as well.
```

### Model config

LLM models can be configured in a taskflow by setting the `model_config` field to the `filekey` of a file of `filetype` `model_config` :

```yaml
seclab-taskflow-agent:
  version: 1
  filetype: taskflow
  filekey: GitHubSecurityLab/seclab-taskflow-agent/taskflows/CVE-2023-2283/CVE-2023-2283

model_config: GitHubSecurityLab/seclab-taskflow-agent/configs/model_config

```

The variables defined in the `model_config` file can then be used throughout the taskflow, e.g.

```yaml
seclab-taskflow-agent:
  version: 1
  filetype: model_config
  filekey: GitHubSecurityLab/seclab-taskflow-agent/configs/model_config
models:
   gpt_latest: gpt-5
```

When `gpt_latest` is used in the taskflow to specify a model, the value `gpt-5` is used:

```yaml
  - task:
      model: gpt_latest
      must_complete: false
      agents:
        - GitHubSecurityLab/seclab-taskflow-agent/personalities/c_auditer
      user_prompt: |

```

This provides a easy way to update model versions in a taskflow.