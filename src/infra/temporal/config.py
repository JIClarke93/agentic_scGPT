"""Shared Temporal configuration for all agents.

This module provides the base configuration for connecting to Temporal servers.
All agent-specific workers should use this config.
"""

from pydantic import BaseModel, Field


class TemporalConfig(BaseModel):
    """Configuration for Temporal server connection.

    This is shared infrastructure - all agents use the same config structure
    but may specify different task queues.

    Attributes:
        host: Temporal server address (default: localhost:7233 for local dev server).
        task_queue: Task queue name for this agent's operations.
        namespace: Temporal namespace (default for local dev server).
    """

    host: str = Field(
        default="localhost:7233",
        description="Temporal server address",
    )
    task_queue: str = Field(
        default="default-tasks",
        description="Task queue name for operations",
    )
    namespace: str = Field(
        default="default",
        description="Temporal namespace",
    )


# Default configuration for local development
DEFAULT_TEMPORAL_CONFIG = TemporalConfig()
