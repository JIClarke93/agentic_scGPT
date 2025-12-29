"""Temporal workflows for scGPT analysis pipelines.

Workflows orchestrate multiple activities with proper retry logic and error handling.
"""

from datetime import timedelta

from temporalio import workflow

from src.infra.temporal import GPU_RETRY_POLICY

from ..constants import COMMON_MARKERS, DEFAULT_BATCH_SIZE
from ..models import (
    AnalysisWorkflowRequest,
    AnalysisWorkflowResult,
    AnnotationRequest,
    AnnotationResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)


@workflow.defn
class ScGPTAnalysisWorkflow:
    """Temporal workflow for complete scGPT analysis pipeline.

    Orchestrates cell type annotation and gene embedding extraction
    with automatic retry on transient failures.
    """

    @workflow.run
    async def run(self, request: AnalysisWorkflowRequest) -> AnalysisWorkflowResult:
        """Run a complete analysis pipeline.

        Args:
            request: Analysis workflow request with expression data path
                and reference dataset configuration.

        Returns:
            AnalysisWorkflowResult with annotation and embedding results.
        """
        # Import activity references for workflow execution
        from ..tools import annotate_cell_types, get_gene_embeddings

        # Step 1: Annotate cell types
        annotation_request = AnnotationRequest(
            expression_data=request.expression_data,
            reference_dataset=request.reference_dataset,
            batch_size=DEFAULT_BATCH_SIZE,
        )

        annotation_result: AnnotationResponse = await workflow.execute_activity(
            annotate_cell_types,
            annotation_request,
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=GPU_RETRY_POLICY,
        )

        # Step 2: Extract embeddings for common marker genes
        embedding_request = EmbeddingRequest(
            gene_list=COMMON_MARKERS,
            model_checkpoint="scGPT_human",
            include_similarity=True,
        )

        embedding_result: EmbeddingResponse = await workflow.execute_activity(
            get_gene_embeddings,
            embedding_request,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=GPU_RETRY_POLICY,
        )

        return AnalysisWorkflowResult(
            annotation=annotation_result,
            embeddings=embedding_result,
            status="complete",
        )
