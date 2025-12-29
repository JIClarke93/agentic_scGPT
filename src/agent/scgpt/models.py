"""Pydantic schemas for MCP tool inputs and outputs.

This module defines the request and response models for all scGPT tools,
following RESTful conventions with Request/Response naming patterns.

Example:
    >>> from src.agent.scgpt.models import AnnotationRequest
    >>> request = AnnotationRequest(expression_data="data/sample.h5ad")
    >>> print(request.batch_size)
    64
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_N_HVG,
    MAX_BATCH_SIZE,
    MAX_N_HVG,
    MIN_BATCH_SIZE,
    MIN_N_HVG,
    MODEL_D_HID,
    MODEL_D_MODEL,
    MODEL_DROPOUT,
    MODEL_N_LAYERS,
    MODEL_NHEAD,
)


class AnnotationRequest(BaseModel):
    """Request schema for cell type annotation.

    Attributes:
        expression_data: Path to h5ad file or CSV containing expression data.
        reference_dataset: Reference dataset for annotation (cellxgene, pbmc, etc.).
        batch_size: Batch size for GPU inference.
    """

    expression_data: str = Field(
        ...,
        description="Path to h5ad file or CSV containing expression data",
    )
    reference_dataset: str = Field(
        default="cellxgene",
        description="Reference dataset for annotation (cellxgene, pbmc, etc.)",
    )
    batch_size: int = Field(
        default=DEFAULT_BATCH_SIZE,
        ge=MIN_BATCH_SIZE,
        le=MAX_BATCH_SIZE,
        description="Batch size for inference",
    )


class CellAnnotation(BaseModel):
    """Single cell annotation result.

    Attributes:
        cell_id: Unique identifier for the cell.
        predicted_type: Predicted cell type label.
        confidence: Confidence score for the prediction (0.0 to 1.0).
        alternative_types: List of alternative cell types with their probabilities.
    """

    cell_id: str
    predicted_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    alternative_types: list[tuple[str, float]] = Field(default_factory=list)


class AnnotationResponse(BaseModel):
    """Response schema for cell type annotation.

    Attributes:
        total_cells: Total number of cells annotated.
        annotations: List of individual cell annotations.
        unique_types: List of unique cell types found in the dataset.
        output_path: Path to the output file with annotations.
    """

    total_cells: int
    annotations: list[CellAnnotation]
    unique_types: list[str]
    output_path: str | None = None


class BatchIntegrationRequest(BaseModel):
    """Request schema for batch integration.

    Attributes:
        dataset_paths: Paths to h5ad files to integrate.
        batch_key: Key in adata.obs identifying batch membership.
        n_hvg: Number of highly variable genes to use for integration.
        output_path: Optional path to save the integrated dataset.
    """

    dataset_paths: list[str] = Field(
        ...,
        min_length=2,
        description="Paths to h5ad files to integrate",
    )
    batch_key: str = Field(
        default="batch",
        description="Key in adata.obs identifying batch",
    )
    n_hvg: int = Field(
        default=DEFAULT_N_HVG,
        ge=MIN_N_HVG,
        le=MAX_N_HVG,
        description="Number of highly variable genes to use",
    )
    output_path: str | None = Field(
        default=None,
        description="Path to save integrated dataset",
    )


class BatchIntegrationResponse(BaseModel):
    """Response schema for batch integration.

    Attributes:
        integrated_path: Path to the integrated dataset file.
        n_cells: Total number of cells in the integrated dataset.
        n_batches: Number of batches that were integrated.
        batch_mixing_score: Quality metric for batch mixing (0.0 to 1.0).
        silhouette_score: Clustering quality metric (-1.0 to 1.0).
    """

    integrated_path: str
    n_cells: int
    n_batches: int
    batch_mixing_score: float = Field(ge=0.0, le=1.0)
    silhouette_score: float = Field(ge=-1.0, le=1.0)


class EmbeddingRequest(BaseModel):
    """Request schema for gene embeddings.

    Attributes:
        gene_list: List of gene symbols to embed.
        model_checkpoint: Model checkpoint to use for embedding extraction.
        include_similarity: Whether to compute pairwise gene similarity matrix.
    """

    gene_list: list[str] = Field(
        ...,
        min_length=1,
        description="List of gene symbols to embed",
    )
    model_checkpoint: str = Field(
        default="scGPT_human",
        description="Model checkpoint to use",
    )
    include_similarity: bool = Field(
        default=False,
        description="Include pairwise gene similarity matrix",
    )


class GeneEmbedding(BaseModel):
    """Single gene embedding representation.

    Attributes:
        gene: Gene symbol (e.g., "TP53", "BRCA1").
        embedding: Vector representation of the gene.
    """

    gene: str
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    """Response schema for gene embeddings.

    Attributes:
        embeddings: List of gene embedding objects.
        embedding_dim: Dimensionality of embedding vectors.
        model_used: Model checkpoint used for embedding extraction.
        similarity_matrix: Optional pairwise cosine similarity matrix.
    """

    embeddings: list[GeneEmbedding]
    embedding_dim: int
    model_used: str
    similarity_matrix: list[list[float]] | None = None


class AlternativeCellType(BaseModel):
    """Alternative cell type prediction with probability score.

    Attributes:
        cell_type: The alternative cell type label.
        probability: Probability score for this cell type (0.0 to 1.0).
    """

    cell_type: str
    probability: float = Field(ge=0.0, le=1.0)


class CellTypePredictionRequest(BaseModel):
    """Request schema for internal cell type prediction.

    Contains the extracted data needed for cell type prediction,
    allowing the prediction logic to work with serializable data
    rather than complex AnnData objects.

    Attributes:
        embedding: Cell embedding vector.
        expression: Expression values for the cell.
        gene_names: Gene names corresponding to expression values.
    """

    embedding: list[float] = Field(..., description="Cell embedding vector")
    expression: list[float] = Field(..., description="Expression values for the cell")
    gene_names: list[str] = Field(..., description="Gene names corresponding to expression values")


class CellTypePredictionResponse(BaseModel):
    """Response schema for internal cell type prediction.

    Attributes:
        predicted_type: Most likely cell type label.
        confidence: Confidence score for the prediction (0.0 to 1.0).
        alternatives: List of alternative cell types with probabilities.
    """

    predicted_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: list[AlternativeCellType] = Field(default_factory=list)


class ScGPTModelConfig(BaseModel):
    """Configuration for scGPT transformer model.

    This config defines the architecture parameters for the scGPT model,
    with sensible defaults matching the pretrained checkpoints.

    Attributes:
        ntoken: Number of tokens in the vocabulary.
        d_model: Model embedding dimension.
        nhead: Number of attention heads.
        d_hid: Hidden layer dimension in feed-forward network.
        nlayers: Number of transformer layers.
        dropout: Dropout rate for regularization.
        pad_token: Token used for padding sequences.
        vocab: Mapping from gene symbols to token indices.
    """

    ntoken: int = Field(..., description="Number of tokens in vocabulary")
    d_model: int = Field(default=MODEL_D_MODEL, description="Model embedding dimension")
    nhead: int = Field(default=MODEL_NHEAD, description="Number of attention heads")
    d_hid: int = Field(default=MODEL_D_HID, description="Hidden layer dimension")
    nlayers: int = Field(default=MODEL_N_LAYERS, description="Number of transformer layers")
    dropout: float = Field(default=MODEL_DROPOUT, ge=0.0, le=1.0, description="Dropout rate")
    pad_token: str = Field(default="<pad>", description="Padding token")
    vocab: dict[str, int] = Field(..., description="Gene vocabulary mapping")


# =============================================================================
# Temporal Workflow Models
# =============================================================================


class AnalysisWorkflowRequest(BaseModel):
    """Request to start an scGPT analysis workflow via Temporal.

    Attributes:
        expression_data: Path to expression data file (.h5ad).
        reference_dataset: Reference dataset for cell type annotation.
    """

    expression_data: str = Field(
        ...,
        description="Path to expression data file (.h5ad)",
    )
    reference_dataset: str = Field(
        default="cellxgene",
        description="Reference dataset for annotation",
    )


class AnalysisWorkflowResult(BaseModel):
    """Result from a completed scGPT analysis workflow.

    Attributes:
        annotation: Cell type annotation results.
        embeddings: Gene embedding extraction results.
        status: Workflow completion status.
    """

    annotation: AnnotationResponse
    embeddings: EmbeddingResponse
    status: Literal["complete", "partial", "failed"] = Field(
        default="complete",
        description="Workflow completion status",
    )


class WorkflowStatus(BaseModel):
    """Status information for a Temporal workflow.

    Attributes:
        workflow_id: Unique identifier for the workflow.
        status: Current workflow status from Temporal.
        task_queue: Task queue the workflow is running on.
    """

    workflow_id: str = Field(..., description="Unique workflow identifier")
    status: str = Field(..., description="Current workflow status")
    task_queue: str = Field(..., description="Task queue name")


class WorkflowResult(BaseModel):
    """Complete result from a Temporal workflow execution.

    Attributes:
        workflow_id: Unique identifier for the workflow.
        status: Final workflow status.
        result: Workflow output data.
    """

    workflow_id: str = Field(..., description="Unique workflow identifier")
    status: str = Field(..., description="Final workflow status")
    result: AnalysisWorkflowResult = Field(..., description="Workflow output data")
