# Temporal Python Sandbox Research

Research on handling Temporal workflow sandbox compatibility issues with common Python dependencies.

## The Problem

Temporal's Python SDK runs workflows in a **sandbox environment** to ensure determinism for replay safety. The sandbox:

1. **Re-imports modules** for every workflow run to isolate global state
2. **Intercepts non-deterministic calls** like `datetime.now()`, `random()`, `os.stat()`
3. **Restricts class extensions** for certain stdlib classes like `threading.local`

This conflicts with many popular Python libraries:

| Library | Issue |
|---------|-------|
| **loguru** | Calls `datetime.now()` at import time |
| **beartype** | Hooks into Python's import system, causes circular imports |
| **pydantic** | Uses beartype internally; compiled version bypasses sandbox type checks |
| **sniffio** | Extends `threading.local` (restricted class) |
| **cryptography/fernet** | PyO3 extension modules get instantiated twice |
| **pandas** | Circular import errors during activity logging |

## Solutions Overview

### 1. Passthrough Modules (Recommended for Most Cases)

Mark modules as "passthrough" so they're not re-imported in the sandbox:

```python
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

restrictions = SandboxRestrictions.default.with_passthrough_modules(
    "pydantic", "loguru", "beartype", "mypackage"
)

worker = Worker(
    client,
    task_queue="my-tasks",
    workflows=[MyWorkflow],
    activities=[my_activity],
    workflow_runner=SandboxedWorkflowRunner(restrictions=restrictions),
)
```

**Pros:**
- Maintains sandbox protections for other code
- Improves performance (modules not reloaded per run)
- Official recommended approach

**Cons:**
- Must identify all problematic modules
- Some modules may still conflict

**Source:** [Temporal Python SDK Sandbox Documentation](https://docs.temporal.io/develop/python/python-sdk-sandbox)

### 2. Import-Time Passthrough

Use context manager in workflow files:

```python
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    import pydantic
    import pandas
    from mypackage import MyModel
```

**Use case:** When you need specific imports passed through only in certain files.

**Source:** [Pydantic AI Temporal Integration](https://ai.pydantic.dev/durable_execution/temporal/)

### 3. Pass Through All Modules

```python
restrictions = SandboxRestrictions.default.with_passthrough_all_modules()
```

**Pros:**
- Solves all import issues at once
- Simple configuration

**Cons:**
- Modules never reloaded per workflow run
- Workflow authors must ensure they don't import non-deterministic modules
- Less safe than targeted passthrough

**Source:** [GitHub Issue #691 - Feature Request for "all passthrough"](https://github.com/temporalio/sdk-python/issues/691)

### 4. Disable Sandbox Entirely (Our Approach)

```python
from temporalio.worker import Worker, UnsandboxedWorkflowRunner

worker = Worker(
    client,
    task_queue="my-tasks",
    workflows=[MyWorkflow],
    activities=[my_activity],
    workflow_runner=UnsandboxedWorkflowRunner(),
)
```

**Pros:**
- Eliminates all sandbox-related errors
- Works with any library combination
- Simple to implement

**Cons:**
- No sandbox protections at all
- Workflow authors must manually ensure determinism
- Not recommended for complex workflows with replay requirements

**When safe to use:**
- Workflows only orchestrate activities (our case)
- All heavy computation happens in activities
- Workflow logic is simple and deterministic

**Source:** [Temporal Python API - UnsandboxedWorkflowRunner](https://python.temporal.io/temporalio.worker.UnsandboxedWorkflowRunner.html)

### 5. Code Separation Pattern

Separate activity definitions from implementations to avoid transitive imports:

```
activities/
├── definitions.py    # Only parameter/result types
└── implementations.py # Actual logic with heavy imports
```

Workflows import only `definitions.py`:

```python
# workflow.py
from activities.definitions import MyActivityParams, MyActivityResult

@workflow.defn
class MyWorkflow:
    @workflow.run
    async def run(self):
        result = await workflow.execute_activity(
            "my_activity",  # String reference, not import
            MyActivityParams(...),
        )
```

**Pros:**
- Workflows don't transitively import activity dependencies
- Maintains sandbox protections
- Good separation of concerns

**Cons:**
- More boilerplate
- Must maintain parallel definition/implementation files

**Source:** [Community Forum - Structuring sandbox friendly activity imports](https://community.temporal.io/t/structuring-sandbox-friendly-activity-imports-when-using-the-python-sdk/6546)

### 6. Move Imports Inside Functions

```python
@activity.defn
async def my_activity():
    import pandas as pd  # Import inside function
    # Use pandas here
```

**Pros:**
- Quick fix for specific activities
- No sandbox configuration needed

**Cons:**
- Repeated import overhead
- Less clean code structure

**Source:** [Community Forum - Structuring sandbox friendly activity imports](https://community.temporal.io/t/structuring-sandbox-friendly-activity-imports-when-using-the-python-sdk/6546)

## Specific Library Issues & Fixes

### Loguru

**Error:**
```
TypeError: _RestrictedProxy.__init__() missing 1 required positional argument: 'matcher'
```

**Cause:** Loguru extends `datetime` class, which is restricted in the sandbox.

**Fix:**
```python
restrictions = SandboxRestrictions.default.with_passthrough_modules("loguru")
```

**Source:** [GitHub Issue #220 - Conflict using temporal-sdk with loguru](https://github.com/temporalio/sdk-python/issues/220)

### Pydantic (Compiled)

**Problem:** Compiled Pydantic bypasses sandbox's `issubclass` proxy, causing datetime fields to be converted to date objects.

**Fix options:**
1. Mark datetime library as passthrough
2. Use non-compiled Pydantic: `pip install pydantic --no-binary :all:`
3. Don't use datetime-based Pydantic fields in workflows

**Source:** [GitHub Issue #207 - SandboxWorkflowRunner Pydantic field types](https://github.com/temporalio/sdk-python/issues/207)

### PyO3/Rust Extensions (cryptography, etc.)

**Problem:** Extension modules get instantiated twice when sandbox re-imports them.

**Fix:**
```python
restrictions = SandboxRestrictions.default.with_passthrough_modules("cryptography")
```

**Source:** [Community Forum - workflow_sandbox imports and re-initializes extension modules](https://community.temporal.io/t/bug-workflow-sandbox-imports-and-re-initizates-extension-modules/10565)

### Pandas

**Problem:** Circular import errors during activity logging.

**Fix:**
```python
with workflow.unsafe.imports_passed_through():
    import pandas
```

**Source:** [Pydantic AI Temporal Documentation](https://ai.pydantic.dev/durable_execution/temporal/)

## Real-World Architecture Patterns

### Pattern 1: Pydantic AI + Temporal

The official Pydantic AI integration uses `TemporalAgent` wrapper:

```python
from pydantic_ai.temporal import TemporalAgent

temporal_agent = TemporalAgent(my_agent)
```

Key requirements:
- Agent `name` field must be stable post-deployment
- All toolsets need explicit `id` parameters
- Dependencies must be Pydantic-serializable

**Source:** [Temporal Blog - Build Durable AI Agents](https://temporal.io/blog/build-durable-ai-agents-pydantic-ai-and-temporal)

### Pattern 2: Video Processing Pipeline

From athiemann.net's vindex project:

```
activities/
├── params.py          # Parameter definitions only
├── frame_extract.py   # Frame extraction implementation
├── transcribe.py      # Audio transcription implementation
└── embed.py           # Embedding computation implementation
```

Key insight: "This avoids callers of the activity to transitively pull in any dependencies of the activity implementation."

**Source:** [Using Temporal with Python](https://www.athiemann.net/2023/01/16/temporal.html)

### Pattern 3: Multi-Agent with Separate Workers

Each agent domain gets its own worker with isolated task queue:

```python
# scGPT worker (GPU tasks)
worker = Worker(client, task_queue="scgpt-tasks", ...)

# FOOGpt worker (CPU tasks)
worker = Worker(client, task_queue="foogpt-tasks", ...)
```

Benefits:
- Independent scaling
- Failure isolation
- Different retry policies per domain

**Source:** Our implementation in this repository

## Best Practices Summary

1. **Keep workflows simple** - Only orchestrate activities, avoid complex logic
2. **Move heavy imports to activities** - Activities run outside the sandbox
3. **Use passthrough for deterministic modules** - Pydantic, dataclasses, your models
4. **Define workflows in clean files** - No side effects, minimal imports
5. **Consider disabling sandbox for simple orchestration** - When workflows are just glue code
6. **Test with sandbox enabled first** - Only disable if you hit actual issues

## Decision Tree

```
Do you need the sandbox?
│
├── YES: Complex workflows with replay requirements
│   │
│   └── Are you hitting import errors?
│       │
│       ├── YES: Try passthrough modules first
│       │   │
│       │   └── Still failing?
│       │       │
│       │       ├── Try code separation pattern
│       │       │
│       │       └── Consider with_passthrough_all_modules()
│       │
│       └── NO: Keep sandbox enabled
│
└── NO: Simple workflows orchestrating activities (our case)
    │
    └── Use UnsandboxedWorkflowRunner()
```

## References

### Official Documentation
- [Temporal Python SDK Sandbox](https://docs.temporal.io/develop/python/python-sdk-sandbox)
- [UnsandboxedWorkflowRunner API](https://python.temporal.io/temporalio.worker.UnsandboxedWorkflowRunner.html)
- [SandboxRestrictions API](https://python.temporal.io/temporalio.worker.workflow_sandbox.SandboxRestrictions.html)

### GitHub Issues
- [#220 - Loguru conflict](https://github.com/temporalio/sdk-python/issues/220)
- [#207 - Pydantic field types](https://github.com/temporalio/sdk-python/issues/207)
- [#450 - pathlib RestrictedWorkflowAccessError](https://github.com/temporalio/sdk-python/issues/450)
- [#691 - Feature request for "all passthrough"](https://github.com/temporalio/sdk-python/issues/691)

### Community Forum
- [Structuring sandbox friendly activity imports](https://community.temporal.io/t/structuring-sandbox-friendly-activity-imports-when-using-the-python-sdk/6546)
- [workflow_sandbox imports and re-initializes extension modules](https://community.temporal.io/t/bug-workflow-sandbox-imports-and-re-initizates-extension-modules/10565)
- [Correct usage of Pydantic in a Temporal Python project](https://community.temporal.io/t/correct-usage-of-pydantic-in-a-temporal-python-project/15896)
- [RestrictedWorkflowAccessError for os.stat](https://community.temporal.io/t/restrictedworkflowaccesserror-cannot-access-os-stat-call-but-i-never-call-os-stat/15702)

### Blog Posts & Articles
- [Temporal Blog - Build Durable AI Agents with Pydantic AI](https://temporal.io/blog/build-durable-ai-agents-pydantic-ai-and-temporal)
- [Pydantic AI - Temporal Integration Guide](https://ai.pydantic.dev/durable_execution/temporal/)
- [Using Temporal with Python (athiemann.net)](https://www.athiemann.net/2023/01/16/temporal.html)
- [Temporal IO Beginner Walkthrough (Medium)](https://morningteofee.medium.com/temporal-beginner-friendly-introduction-2bf9f084e0ec)
- [Temporal Python 1.0.0 – A Durable, Distributed Asyncio Event Loop (Dev.to)](https://dev.to/temporalio/temporal-python-100-a-durable-distributed-asyncio-event-loop-5hjh)
