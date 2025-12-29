# Temporal Orchestration for scGPT

Local Temporal setup for reliable scGPT workflow orchestration with automatic retries.

## Quick Start

### 1. Install Temporal CLI (Free, Local)

```bash
# Windows (winget)
winget install temporalio.cli

# macOS (brew)
brew install temporal

# Or download from: https://docs.temporal.io/cli#install
```

### 2. Start Local Temporal Server

```bash
temporal server start-dev
```

This starts a **free local server** at `localhost:7233` with a web UI at `http://localhost:8233`.

### 3. Run the Worker

```bash
uv run python -m src.agent.scgpt.temporal.worker
```

### 4. Start a Workflow

```python
import asyncio
from src.agent.scgpt.temporal import ScGPTTemporalClient
from src.agent.scgpt.models import AnalysisWorkflowRequest

async def main():
    client = ScGPTTemporalClient()
    await client.connect()

    request = AnalysisWorkflowRequest(
        expression_data="data/sample.h5ad",
        reference_dataset="cellxgene",
    )

    workflow_id = await client.start_analysis_workflow(request)
    print(f"Started workflow: {workflow_id}")

    # Wait for result
    result = await client.get_workflow_result(workflow_id)
    print(f"Found {len(result.result.annotation.unique_types)} cell types")

asyncio.run(main())
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ScGPTTemporalClient                       │
│                  (Start/Query Workflows)                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               Local Temporal Server                          │
│              (localhost:7233 - FREE)                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      ScGPTWorker                             │
│                (Polls & Executes Tasks)                      │
│                                                              │
│  Activities (from src.agent.scgpt.tools):                   │
│  ├── annotate_cell_types    @activity.defn                  │
│  ├── integrate_batches      @activity.defn                  │
│  └── get_gene_embeddings    @activity.defn                  │
└─────────────────────────────────────────────────────────────┘
```

## Multi-Agent Architecture

### One Worker Per Agent Domain

Each agent (scGPT, FOOGpt, etc.) gets its **own worker** with its **own task queue**:

```
src/agent/
├── scgpt/
│   ├── tools/              ← Activities with @activity.defn
│   │   ├── annotate.py
│   │   ├── integrate.py
│   │   └── embeddings.py
│   └── temporal/
│       ├── worker.py       ← ScGPTWorker (queue: "scgpt-tasks")
│       ├── workflows.py
│       └── client.py
│
├── foogpt/                  ← Future agent example
│   ├── tools/
│   │   └── foo_activity.py
│   └── temporal/
│       ├── worker.py       ← FOOGptWorker (queue: "foogpt-tasks")
│       ├── workflows.py
│       └── client.py
```

### Why Separate Workers?

| Benefit | Description |
|---------|-------------|
| **Independent scaling** | scGPT needs GPU, FOOGpt maybe CPU-only |
| **Independent deployment** | Update one agent without touching others |
| **Failure isolation** | scGPT crashes, FOOGpt keeps running |
| **Clear ownership** | Each team owns their worker |
| **Different retry policies** | GPU tasks vs API calls have different needs |

### Running Multiple Workers

```bash
# Terminal 1: scGPT worker (GPU tasks)
uv run python -m src.agent.scgpt.temporal.worker

# Terminal 2: FOOGpt worker (if you add it later)
uv run python -m src.agent.foogpt.temporal.worker
```

Both workers connect to the **same Temporal server** but poll **different task queues**.

### When to Share a Worker

You might use ONE worker for multiple activity types when:
- Activities have identical resource requirements
- They're tightly coupled and always deployed together
- Simplicity outweighs isolation benefits

The `ScGPTWorker` already does this - it handles `annotate`, `integrate`, and `embed` because they all need GPU and belong to the same domain.

## Module Structure

```
src/agent/scgpt/temporal/
├── __init__.py      # Public exports
├── worker.py        # ScGPTWorker - runs activities
├── workflows.py     # ScGPTAnalysisWorkflow - orchestrates activities
├── client.py        # ScGPTTemporalClient - starts workflows
└── README.md        # This file

Models live in src/agent/scgpt/models.py:
├── TemporalConfig           # Server connection settings
├── AnalysisWorkflowRequest  # Workflow input
├── AnalysisWorkflowResult   # Workflow output
├── WorkflowStatus           # Current state
└── WorkflowResult           # Final result with metadata
```

## Adding a New Agent with Temporal

1. **Create the agent structure:**
   ```
   src/agent/newagent/
   ├── tools/
   │   └── my_activity.py    # Add @activity.defn decorator
   ├── models.py             # Request/Response models
   └── temporal/
       ├── __init__.py
       ├── worker.py         # NewAgentWorker
       ├── workflows.py      # NewAgentWorkflow
       └── client.py         # NewAgentTemporalClient
   ```

2. **Define activities in tools with `@activity.defn`:**
   ```python
   from temporalio import activity

   @activity.defn
   async def my_activity(request: MyRequest) -> MyResponse:
       # Your logic here
       pass
   ```

3. **Create worker with unique task queue:**
   ```python
   # temporal/worker.py
   from ..tools import my_activity

   @dataclass
   class NewAgentWorker:
       config: TemporalConfig = field(default_factory=lambda: TemporalConfig(
           task_queue="newagent-tasks"  # Unique queue name!
       ))
       # ... rest same as ScGPTWorker
   ```

4. **Run the worker:**
   ```bash
   uv run python -m src.agent.newagent.temporal.worker
   ```

## Retry Policy

GPU operations use this retry policy:

```python
GPU_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=5),
    maximum_interval=timedelta(minutes=2),
    backoff_coefficient=2.0,
    maximum_attempts=3,
    non_retryable_error_types=[
        "InvalidDataError",      # Bad input, won't help to retry
        "ModelNotFoundError",    # Missing checkpoint
        "FileNotFoundError",     # Missing data file
    ],
)
```

## Web UI

Access the Temporal web UI at `http://localhost:8233` to:
- View running workflows
- Inspect workflow history
- Debug failed activities
- Terminate stuck workflows

## Why Temporal?

- **Automatic retries**: GPU OOM? Network blip? Temporal retries with backoff
- **Durability**: Workflows survive worker restarts
- **Visibility**: Web UI shows exactly what's happening
- **Free**: Local dev server costs nothing
- **Multi-agent ready**: Each agent gets isolated workers

### Note: Replay vs Caching

Temporal stores activity results for **replay**, not for **memoization**. This is an important distinction:

```
First execution:
  Workflow starts
  → annotate_cell_types executes (GPU runs, 5 min) → result stored in history
  → get_gene_embeddings executes (GPU runs, 3 min) → result stored in history
  → Workflow completes

After crash + replay:
  Workflow replays from history
  → annotate_cell_types: result read from history (not re-executed!)
  → get_gene_embeddings: result read from history (not re-executed!)
  → Workflow completes instantly
```

**What Temporal does:**
- Stores completed activity results in workflow history
- On replay (after crash/restart), skips re-execution and uses stored results
- Ensures workflows can resume from exactly where they left off

**What Temporal does NOT do:**
- Cache results across different workflow runs
- Skip execution if you call the same activity with the same inputs twice
- Provide application-level memoization

If you want true memoization (same inputs → skip GPU work), implement it in your activity:

```python
@activity.defn
async def annotate_cell_types(request: AnnotationRequest) -> AnnotationResponse:
    cache_key = hash_request(request)
    if cached := check_cache(cache_key):
        return cached  # Skip GPU work

    result = run_inference(request)
    store_cache(cache_key, result)
    return result
```

**Why the sandbox matters for replay:**

The sandbox ensures workflow logic is deterministic so replay reaches the same branching decisions:

```python
# BAD - non-deterministic, breaks replay
if datetime.now().hour < 12:
    await workflow.execute_activity(morning_task, ...)

# GOOD - deterministic, replay-safe
if request.use_fast_mode:
    await workflow.execute_activity(fast_task, ...)
```

On replay, Temporal needs to follow the same code path to match stored results with the right activity calls.

### Deterministic Alternatives

Temporal provides deterministic versions of common non-deterministic operations:

```python
from temporalio import workflow

@workflow.defn
class MyWorkflow:
    @workflow.run
    async def run(self):
        # ❌ BAD - non-deterministic, breaks replay
        now = datetime.now()
        random_val = random.random()
        my_uuid = uuid.uuid4()
        await asyncio.sleep(60)

        # ✅ GOOD - Temporal's deterministic versions
        now = workflow.now()              # Same value on replay
        random_val = workflow.random()    # Same value on replay
        my_uuid = workflow.uuid4()        # Same value on replay
        await workflow.sleep(60)          # Durable timer, survives restarts
```

| Operation | Problem | Temporal Solution |
|-----------|---------|-------------------|
| `datetime.now()` | Different on replay | `workflow.now()` - returns workflow's logical time |
| `random.random()` | Different on replay | `workflow.random()` - seeded deterministically |
| `uuid.uuid4()` | Different on replay | `workflow.uuid4()` - deterministic generation |
| `asyncio.sleep()` | Lost on crash | `workflow.sleep()` - durable, resumes after restart |

**Durable timers example:**

```python
@workflow.defn
class ReminderWorkflow:
    @workflow.run
    async def run(self, request: ReminderRequest):
        created_at = workflow.now()

        # Sleep for 24 hours - survives worker restarts!
        await workflow.sleep(timedelta(hours=24))

        # Runs exactly 24h later, even if worker crashed mid-sleep
        await workflow.execute_activity(send_reminder, request)
```

If the worker crashes 12h in, when it restarts Temporal replays the workflow, sees the timer was started 12h ago, waits only the remaining 12h, then executes the activity.

**Activities can use real datetime:**

Activities run outside the sandbox and aren't replayed (only their results are stored), so real `datetime.now()` is fine:

```python
@activity.defn
async def annotate_cell_types(request: AnnotationRequest) -> AnnotationResponse:
    started_at = datetime.now()  # Fine - activities aren't replayed
    result = run_inference(request)
    logger.info(f"Took {datetime.now() - started_at}")
    return result
```

## Lessons Learned

### Sandbox Compatibility with Dependencies

**Problem:** Temporal runs workflows in a sandbox that intercepts imports and blocks non-deterministic operations for replay safety. However, our dependencies cause conflicts:

1. `beartype` - Hooks into Python's import system → circular import errors
2. `loguru` - Calls `datetime.now()` at import time → blocked by sandbox
3. `pydantic` - Uses beartype internally

**Solution:** Disable the sandbox entirely:

```python
from temporalio.worker.workflow_sandbox import UnsandboxedWorkflowRunner

Worker(
    client,
    task_queue="...",
    workflows=[...],
    activities=[...],
    workflow_runner=UnsandboxedWorkflowRunner(),  # Disable sandbox
)
```

**Why this is safe for scGPT:**
- Our workflows only **orchestrate** activities (call them in sequence, handle results)
- All heavy computation (GPU inference, data processing) happens in **activities**
- Activities run **outside** the sandbox anyway
- The workflow code itself is simple and deterministic

**When you DO need the sandbox:**
- Workflows with complex logic that could be non-deterministic
- Long-running workflows that need replay guarantees
- Production environments with strict determinism requirements

**Partial fix (if you need some sandboxing):**
```python
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

sandbox_runner = SandboxedWorkflowRunner(
    restrictions=SandboxRestrictions.default.with_passthrough_modules("beartype", "loguru")
)
```
But this may not catch all problematic imports.
