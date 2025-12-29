"""Temporal client for interacting with scGPT workflows.

Provides utilities to start, query, and manage workflows running on the local Temporal server.
"""

import uuid
from dataclasses import dataclass, field

from loguru import logger
from temporalio.client import Client

from src.infra.temporal import TemporalConfig

from ..models import (
    AnalysisWorkflowRequest,
    AnalysisWorkflowResult,
    WorkflowResult,
    WorkflowStatus,
)
from .worker import SCGPT_TEMPORAL_CONFIG
from .workflows import ScGPTAnalysisWorkflow


@dataclass
class ScGPTTemporalClient:
    """Client for interacting with scGPT Temporal workflows.

    Attributes:
        config: Temporal connection configuration.
        client: Connected Temporal client (set after connect).
    """

    config: TemporalConfig = field(default_factory=lambda: SCGPT_TEMPORAL_CONFIG)
    client: Client | None = field(default=None, init=False)

    async def connect(self) -> None:
        """Connect to the local Temporal server."""
        logger.info(f"Connecting to Temporal at {self.config.host}")
        self.client = await Client.connect(
            self.config.host,
            namespace=self.config.namespace,
        )

    async def start_analysis_workflow(
        self,
        request: AnalysisWorkflowRequest,
        workflow_id: str | None = None,
    ) -> str:
        """Start a new scGPT analysis workflow.

        Args:
            request: Analysis workflow request with expression data and options.
            workflow_id: Optional workflow ID (auto-generated if not provided).

        Returns:
            Workflow ID for tracking.
        """
        if self.client is None:
            await self.connect()

        assert self.client is not None

        if workflow_id is None:
            workflow_id = f"scgpt-analysis-{uuid.uuid4().hex[:8]}"

        logger.info(f"Starting workflow {workflow_id} for {request.expression_data}")

        handle = await self.client.start_workflow(
            ScGPTAnalysisWorkflow.run,
            request,
            id=workflow_id,
            task_queue=self.config.task_queue,
        )

        logger.info(f"Workflow started with ID: {handle.id}")
        return handle.id

    async def get_workflow_result(self, workflow_id: str) -> WorkflowResult:
        """Get the result of a completed workflow.

        Args:
            workflow_id: ID of the workflow to query.

        Returns:
            WorkflowResult with status and typed output data.
        """
        if self.client is None:
            await self.connect()

        assert self.client is not None

        handle = self.client.get_workflow_handle(workflow_id)
        result: AnalysisWorkflowResult = await handle.result()
        desc = await handle.describe()

        status_name = desc.status.name if desc.status else "UNKNOWN"

        return WorkflowResult(
            workflow_id=workflow_id,
            status=status_name,
            result=result,
        )

    async def get_workflow_status(self, workflow_id: str) -> WorkflowStatus:
        """Get the current status of a workflow.

        Args:
            workflow_id: ID of the workflow to query.

        Returns:
            WorkflowStatus with current state information.
        """
        if self.client is None:
            await self.connect()

        assert self.client is not None

        handle = self.client.get_workflow_handle(workflow_id)
        desc = await handle.describe()

        status_name = desc.status.name if desc.status else "UNKNOWN"

        return WorkflowStatus(
            workflow_id=workflow_id,
            status=status_name,
            task_queue=self.config.task_queue,
        )

    async def cancel_workflow(self, workflow_id: str) -> None:
        """Cancel a running workflow.

        Args:
            workflow_id: ID of the workflow to cancel.
        """
        if self.client is None:
            await self.connect()

        assert self.client is not None

        handle = self.client.get_workflow_handle(workflow_id)
        await handle.cancel()
        logger.info(f"Workflow {workflow_id} cancelled")


async def run_analysis(
    request: AnalysisWorkflowRequest,
    wait_for_result: bool = True,
    config: TemporalConfig | None = None,
) -> WorkflowResult | str:
    """Convenience function to run an scGPT analysis via Temporal.

    Args:
        request: Analysis workflow request with expression data and options.
        wait_for_result: If True, wait and return result; if False, return workflow ID.
        config: Temporal configuration (uses defaults if not provided).

    Returns:
        WorkflowResult if wait_for_result=True, else workflow ID string.
    """
    client = ScGPTTemporalClient(config=config or SCGPT_TEMPORAL_CONFIG)
    await client.connect()

    workflow_id = await client.start_analysis_workflow(request=request)

    if wait_for_result:
        return await client.get_workflow_result(workflow_id)

    return workflow_id
