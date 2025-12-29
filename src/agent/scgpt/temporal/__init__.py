"""Temporal orchestration for scGPT workflows.

This module provides:
- ScGPTWorker: Worker that runs activities and workflows
- ScGPTTemporalClient: Client for starting and managing workflows
- ScGPTAnalysisWorkflow: Multi-step analysis workflow
- TemporalConfig: Configuration for local Temporal server

The activities (annotate_cell_types, integrate_batches, get_gene_embeddings)
are defined in src.agent.scgpt.tools with @activity.defn decorators.

Usage:
    # Start local Temporal server first:
    # temporal server start-dev

    # Then run the worker:
    from src.agent.scgpt.temporal import ScGPTWorker
    worker = ScGPTWorker()
    await worker.run()

    # Start a workflow from client:
    from src.agent.scgpt.temporal import ScGPTTemporalClient
    from src.agent.scgpt.models import AnalysisWorkflowRequest

    client = ScGPTTemporalClient()
    await client.connect()
    request = AnalysisWorkflowRequest(expression_data="data/sample.h5ad")
    workflow_id = await client.start_analysis_workflow(request)
"""

from src.infra.temporal import GPU_RETRY_POLICY, TemporalConfig

from .client import ScGPTTemporalClient, run_analysis
from .worker import SCGPT_TEMPORAL_CONFIG, ScGPTWorker, run_worker
from .workflows import ScGPTAnalysisWorkflow

__all__ = [
    # Config (from shared infra)
    "TemporalConfig",
    # scGPT-specific config
    "SCGPT_TEMPORAL_CONFIG",
    # Worker
    "ScGPTWorker",
    "run_worker",
    # Client
    "ScGPTTemporalClient",
    "run_analysis",
    # Workflow
    "ScGPTAnalysisWorkflow",
    # Policies (from shared infra)
    "GPU_RETRY_POLICY",
]
