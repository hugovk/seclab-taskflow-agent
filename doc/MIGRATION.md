# Jinja2 Templating Migration Guide

This guide explains how to migrate taskflow YAML files from the old custom template syntax to Jinja2 templating with version `"1.0"` format.

## Overview

The new version replaces the custom regex-based template processing with Jinja2, providing:
- More powerful templating features (filters, conditionals, loops)
- Better error messages with clear variable undefined errors
- Industry-standard syntax familiar to many developers
- Extensibility for future template features
- String-based version format (e.g., `"1.0"`) for semantic versioning support

## Syntax Changes

### 1. Global Variables

**Old syntax:**
```yaml
globals:
  fruit: apples
taskflow:
  - task:
      user_prompt: |
        Tell me about {{ GLOBALS_fruit }}.
```

**New syntax:**
```yaml
globals:
  fruit: apples
taskflow:
  - task:
      user_prompt: |
        Tell me about {{ globals.fruit }}.
```

**Nested structures:**
```yaml
globals:
  config:
    model: gpt-4
    temperature: 0.7
taskflow:
  - task:
      user_prompt: |
        Using {{ globals.config.model }} with temp {{ globals.config.temperature }}
```

### 2. Input Variables

**Old syntax:**
```yaml
user_prompt: |
  Color: {{ INPUTS_color }}
```

**New syntax:**
```yaml
user_prompt: |
  Color: {{ inputs.color }}
```

### 3. Result Variables

**Old syntax (primitives):**
```yaml
repeat_prompt: true
user_prompt: |
  Process {{ RESULT }}
```

**New syntax:**
```yaml
repeat_prompt: true
user_prompt: |
  Process {{ result }}
```

**Old syntax (dictionary keys):**
```yaml
user_prompt: |
  Function {{ RESULT_name }} has body {{ RESULT_body }}
```

**New syntax:**
```yaml
user_prompt: |
  Function {{ result.name }} has body {{ result.body }}
```

### 4. Environment Variables

**Old syntax:**
```yaml
env:
  DATABASE: "{{ env DATABASE_URL }}"
```

**New syntax:**
```yaml
env:
  DATABASE: "{{ env('DATABASE_URL') }}"
```

**With defaults (new feature):**
```yaml
env:
  DATABASE: "{{ env('DATABASE_URL', 'localhost:5432') }}"
```

### 5. Reusable Prompts

**Old syntax:**
```yaml
user_prompt: |
  Main task.
  {{ PROMPTS_examples.prompts.shared }}
```

**New syntax:**
```yaml
user_prompt: |
  Main task.
  {% include 'examples.prompts.shared' %}
```

## New Jinja2 Features

### Filters

Transform values with filters:

```yaml
user_prompt: |
  Uppercase: {{ globals.name | upper }}
  Lowercase: {{ globals.name | lower }}
  Default: {{ globals.optional | default('N/A') }}
  List length: {{ globals.items | length }}
```

### Conditionals

Add conditional logic:

```yaml
user_prompt: |
  {% if globals.debug_mode %}
  Running in debug mode
  {% else %}
  Running in production mode
  {% endif %}

  {% if result.score > 0.8 %}
  High confidence result
  {% endif %}
```

### Loops

Iterate over collections:

```yaml
user_prompt: |
  Analyze these functions:
  {% for func in result.functions %}
  - {{ func.name }}: {{ func.complexity }}
  {% endfor %}
```

### Math Operations

Perform calculations:

```yaml
user_prompt: |
  Sum: {{ result.a + result.b }}
  Product: {{ result.count * 2 }}
  Comparison: {% if result.score > 0.5 %}Pass{% else %}Fail{% endif %}
```

## Automated Migration

Use the provided migration script:

```bash
# Migrate all YAML files in directory
python scripts/migrate_to_jinja2.py /path/to/taskflows

# Preview changes without writing
python scripts/migrate_to_jinja2.py --dry-run /path/to/taskflows

# Migrate specific file
python scripts/migrate_to_jinja2.py myflow.yaml
```

## Manual Migration Checklist

1. Update YAML version to `"1.0"` (string format, e.g., `version: "1.0"`)
2. Replace `{{ GLOBALS_` with `{{ globals.`
3. Replace `{{ INPUTS_` with `{{ inputs.`
4. Replace `{{ RESULT_` with `{{ result.`
5. Replace `{{ RESULT }}` with `{{ result }}`
6. Replace `{{ env VAR }}` with `{{ env('VAR') }}`
7. Replace `{{ PROMPTS_` with `{% include '` and add closing `' %}`
8. Test taskflow execution

## Testing Your Migration

```bash
# Run specific taskflow
python -m seclab_taskflow_agent -t your.taskflow.name

# Run with globals
python -m seclab_taskflow_agent -t your.taskflow.name -g key=value
```

## Common Issues

### Issue: `UndefinedError: 'globals' is undefined`

**Cause:** Using `{{ globals.key }}` when no globals are defined

**Fix:** Either define globals in taskflow or use Jinja2's get method:
```yaml
{{ globals.get('key', 'default') }}
```

### Issue: `TemplateNotFound: examples.prompts.mypromp`

**Cause:** Typo in include path

**Fix:** Verify path matches file location exactly

### Issue: Environment variable errors

**Cause:** Required env var not set

**Fix:** Set env var or make it optional:
```yaml
{{ env('VAR', 'default') }}
```

## Backwards Compatibility

### Version Format

The system now uses string-based semantic versioning (e.g., `"1.0"`, `"1.1"`). For backwards compatibility:

- Integer `version: 1` is automatically converted to `"1.0"`
- Float `version: 1.2` is automatically converted to `"1.2"`
- String versions like `version: "1.0"` are used as-is

Only versions in the `1.x` series are supported. Any version that doesn't convert to `"1.x"` format will be rejected:

```
VersionException: Unsupported version: <version>. Only version 1.x is supported.
```

### Template Syntax

The old custom template syntax (e.g., `{{ GLOBALS_key }}`, `{{ INPUTS_key }}`) is **no longer supported**. All files using the old syntax must be migrated to Jinja2 syntax using the migration script:

```bash
python scripts/migrate_to_jinja2.py <file>
```

The migration script will also update your version format to the string-based `"1.0"` format.

## Additional Resources

- [Jinja2 Documentation](https://jinja.palletsprojects.com/)
- [Jinja2 Template Designer Documentation](https://jinja.palletsprojects.com/en/3.1.x/templates/)
- Example taskflows in `examples/taskflows/`
