"""Temporal activities wrapping scGPT operations with retry logic."""

from datetime import timedelta

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

from src.agent.scgpt.constants import COMMON_MARKERS, DEFAULT_BATCH_SIZE, DEFAULT_N_HVG
from src.agent.scgpt.models import AnnotationRequest, BatchIntegrationRequest, EmbeddingRequest
from src.agent.scgpt.tools import annotate_cell_types, get_gene_embeddings, integrate_batches


# Custom exception for non-retryable errors
class InvalidDataError(Exception):
    """Raised when input data is invalid and retry won't help."""

    pass


class ModelNotFoundError(Exception):
    """Raised when model checkpoint is not found."""

    pass


# Define retry policy for GPU operations
GPU_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=5),
    maximum_interval=timedelta(minutes=2),
    backoff_coefficient=2.0,
    maximum_attempts=3,
    non_retryable_error_types=[
        "InvalidDataError",
        "ModelNotFoundError",
        "FileNotFoundError",
    ],
)


@activity.defn
async def annotate_cells_activity(
    expression_data: str,
    reference_dataset: str = "cellxgene",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """
    Temporal activity for cell type annotation with automatic retry.

    This activity wraps the annotate_cell_types tool with Temporal's
    retry capabilities, automatically retrying on transient failures
    like GPU OOM errors.

    Args:
        expression_data: Path to expression data
        reference_dataset: Reference dataset name
        batch_size: Batch size for inference

    Returns:
        Annotation results as dictionary
    """
    activity.logger.info(f"Starting cell annotation for {expression_data}")

    try:
        request = AnnotationRequest(
            expression_data=expression_data,
            reference_dataset=reference_dataset,
            batch_size=batch_size,
        )
        result = await annotate_cell_types(request)
        activity.logger.info(f"Annotated {result.total_cells} cells")
        return result.model_dump()

    except FileNotFoundError:
        # Don't retry if file doesn't exist
        raise
    except MemoryError:
        # GPU OOM - will be retried with exponential backoff
        activity.logger.warning("GPU OOM error, will retry with backoff")
        raise
    except Exception as e:
        activity.logger.error(f"Annotation failed: {e}")
        raise


@activity.defn
async def integrate_batches_activity(
    dataset_paths: list[str],
    batch_key: str = "batch",
    n_hvg: int = DEFAULT_N_HVG,
    output_path: str | None = None,
) -> dict:
    """
    Temporal activity for batch integration with automatic retry.

    Args:
        dataset_paths: Paths to datasets to integrate
        batch_key: Batch identifier key
        n_hvg: Number of highly variable genes
        output_path: Output path for integrated data

    Returns:
        Integration results as dictionary
    """
    activity.logger.info(f"Starting batch integration of {len(dataset_paths)} datasets")

    try:
        request = BatchIntegrationRequest(
            dataset_paths=dataset_paths,
            batch_key=batch_key,
            n_hvg=n_hvg,
            output_path=output_path,
        )
        result = await integrate_batches(request)
        activity.logger.info(f"Integration complete: {result.n_cells} cells")
        return result.model_dump()

    except FileNotFoundError:
        raise
    except Exception as e:
        activity.logger.error(f"Integration failed: {e}")
        raise


@activity.defn
async def embed_genes_activity(
    gene_list: list[str],
    model_checkpoint: str = "scGPT_human",
    include_similarity: bool = False,
) -> dict:
    """
    Temporal activity for gene embedding extraction with automatic retry.

    Args:
        gene_list: Genes to embed
        model_checkpoint: Model checkpoint to use
        include_similarity: Whether to compute similarity matrix

    Returns:
        Embedding results as dictionary
    """
    activity.logger.info(f"Extracting embeddings for {len(gene_list)} genes")

    try:
        request = EmbeddingRequest(
            gene_list=gene_list,
            model_checkpoint=model_checkpoint,
            include_similarity=include_similarity,
        )
        result = await get_gene_embeddings(request)
        activity.logger.info(f"Extracted {len(result.embeddings)} embeddings")
        return result.model_dump()

    except Exception as e:
        activity.logger.error(f"Embedding extraction failed: {e}")
        raise


# Example workflow using the activities
@workflow.defn
class ScGPTAnalysisWorkflow:
    """
    Temporal workflow for complete scGPT analysis pipeline.

    This workflow orchestrates multiple scGPT operations with proper
    retry logic and error handling.
    """

    @workflow.run
    async def run(
        self,
        expression_data: str,
        reference_dataset: str = "cellxgene",
    ) -> dict:
        """
        Run a complete analysis pipeline.

        Args:
            expression_data: Path to expression data
            reference_dataset: Reference dataset for annotation

        Returns:
            Combined results from all analysis steps
        """
        # Step 1: Annotate cell types
        annotation_result = await workflow.execute_activity(
            annotate_cells_activity,
            args=[expression_data, reference_dataset, DEFAULT_BATCH_SIZE],
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=GPU_RETRY_POLICY,
        )

        # Step 2: Extract embeddings for marker genes
        embedding_result = await workflow.execute_activity(
            embed_genes_activity,
            args=[COMMON_MARKERS, "scGPT_human", True],
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=GPU_RETRY_POLICY,
        )

        return {
            "annotation": annotation_result,
            "embeddings": embedding_result,
            "status": "complete",
        }
