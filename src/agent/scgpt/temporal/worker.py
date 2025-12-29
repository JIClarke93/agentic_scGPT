"""Temporal worker for running scGPT activities and workflows.

This worker connects to a LOCAL Temporal server (no cloud/paid service needed).
Start the local Temporal server with: temporal server start-dev
"""

import asyncio
from dataclasses import dataclass, field

from loguru import logger
from temporalio.client import Client
from temporalio.worker import Worker

# NOTE: We disable the workflow sandbox because our dependencies (loguru, beartype, pydantic)
# use non-deterministic code at import time (datetime.now, import hooks, etc.).
# This is safe for our use case since workflows only orchestrate activities,
# and all heavy computation happens in activities (which run outside the sandbox).
from temporalio.worker.workflow_sandbox._runner import UnsandboxedWorkflowRunner

from src.infra.temporal import TemporalConfig

from ..tools import (
    annotate_cell_types,
    get_gene_embeddings,
    integrate_batches,
)

from .workflows import ScGPTAnalysisWorkflow


# scGPT-specific configuration with its own task queue
SCGPT_TEMPORAL_CONFIG = TemporalConfig(task_queue="scgpt-tasks")


@dataclass
class ScGPTWorker:
    """Temporal worker for scGPT activities and workflows.

    Attributes:
        config: Temporal connection configuration.
        client: Connected Temporal client (set after connect).
        worker: Temporal worker instance (set after connect).
    """

    config: TemporalConfig = field(default_factory=lambda: SCGPT_TEMPORAL_CONFIG)
    client: Client | None = field(default=None, init=False)
    worker: Worker | None = field(default=None, init=False)

    async def connect(self) -> None:
        """Connect to the local Temporal server and initialize worker."""
        logger.info(f"Connecting to Temporal server at {self.config.host}")
        self.client = await Client.connect(
            self.config.host,
            namespace=self.config.namespace,
        )

        self.worker = Worker(
            self.client,
            task_queue=self.config.task_queue,
            workflows=[ScGPTAnalysisWorkflow],
            activities=[
                annotate_cell_types,
                integrate_batches,
                get_gene_embeddings,
            ],
            workflow_runner=UnsandboxedWorkflowRunner(),  # Disable sandbox (see note above)
        )
        logger.info(f"Worker initialized on task queue '{self.config.task_queue}'")

    async def run(self) -> None:
        """Run the worker (blocking). Call connect() first."""
        if self.worker is None:
            await self.connect()

        assert self.worker is not None
        logger.info("Starting worker - Press Ctrl+C to stop")
        await self.worker.run()


async def run_worker(config: TemporalConfig | None = None) -> None:
    """
    Run the Temporal worker (blocking).

    Args:
        config: Temporal configuration (uses defaults if not provided)
    """
    worker = ScGPTWorker(config=config or SCGPT_TEMPORAL_CONFIG)
    await worker.connect()
    await worker.run()


def main() -> None:
    """Entry point for running the worker."""
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")


if __name__ == "__main__":
    main()
