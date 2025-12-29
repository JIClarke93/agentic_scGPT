"""Shared Temporal infrastructure for all agents.

This module provides common infrastructure that all agent-specific
Temporal workers and clients can use:

- TemporalConfig: Connection configuration
- Retry policies: GPU_RETRY_POLICY, API_RETRY_POLICY, etc.

Usage:
    from src.infra.temporal import TemporalConfig, GPU_RETRY_POLICY

    # Create agent-specific config
    config = TemporalConfig(task_queue="my-agent-tasks")

    # Use shared retry policies in workflows
    await workflow.execute_activity(
        my_activity,
        args=[...],
        retry_policy=GPU_RETRY_POLICY,
    )
"""

from .config import DEFAULT_TEMPORAL_CONFIG, TemporalConfig
from .policies import (
    API_RETRY_POLICY,
    GPU_RETRY_POLICY,
    IO_RETRY_POLICY,
    NO_RETRY_POLICY,
)

__all__ = [
    # Config
    "TemporalConfig",
    "DEFAULT_TEMPORAL_CONFIG",
    # Policies
    "GPU_RETRY_POLICY",
    "API_RETRY_POLICY",
    "IO_RETRY_POLICY",
    "NO_RETRY_POLICY",
]
