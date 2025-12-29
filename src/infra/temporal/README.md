# Shared Temporal Infrastructure

Common Temporal infrastructure shared across all agents.

## What's Here

```
src/infra/temporal/
├── __init__.py     # Public exports
├── config.py       # TemporalConfig - connection settings
├── policies.py     # Shared retry policies
└── README.md       # This file
```

## Usage

All agents import shared infrastructure from here:

```python
from src.infra.temporal import TemporalConfig, GPU_RETRY_POLICY

# Create agent-specific config with custom task queue
config = TemporalConfig(task_queue="my-agent-tasks")

# Use shared retry policies
await workflow.execute_activity(
    my_activity,
    args=[...],
    retry_policy=GPU_RETRY_POLICY,
)
```

## Available Exports

### TemporalConfig

Base configuration for Temporal server connection:

```python
class TemporalConfig(BaseModel):
    host: str = "localhost:7233"      # Server address
    task_queue: str = "default-tasks" # Override per agent!
    namespace: str = "default"        # Temporal namespace
```

### Retry Policies

| Policy | Use Case |
|--------|----------|
| `GPU_RETRY_POLICY` | GPU operations (OOM, CUDA errors) |
| `API_RETRY_POLICY` | External API calls |
| `IO_RETRY_POLICY` | File I/O operations |
| `NO_RETRY_POLICY` | Operations that should not retry |

## Agent-Specific Code

Agent-specific Temporal code lives in each agent's module:

```
src/agent/
├── scgpt/temporal/     # scGPT-specific workers, workflows, clients
├── foogpt/temporal/    # FOOGpt-specific (future)
```

Each agent:
1. Imports `TemporalConfig` from here
2. Creates its own config with a unique `task_queue`
3. Defines its own worker, workflows, and client

See `src/agent/scgpt/temporal/README.md` for a complete example.
