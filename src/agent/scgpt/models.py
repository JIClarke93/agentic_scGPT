"""Pydantic schemas for MCP tool inputs and outputs."""

from pydantic import BaseModel, Field


class AnnotationRequest(BaseModel):
    """Request schema for cell type annotation."""

    expression_data: str = Field(
        ...,
        description="Path to h5ad file or CSV containing expression data",
    )
    reference_dataset: str = Field(
        default="cellxgene",
        description="Reference dataset for annotation (cellxgene, pbmc, etc.)",
    )
    batch_size: int = Field(
        default=64,
        ge=1,
        le=512,
        description="Batch size for inference",
    )


class CellAnnotation(BaseModel):
    """Single cell annotation result."""

    cell_id: str
    predicted_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    alternative_types: list[tuple[str, float]] = Field(default_factory=list)


class AnnotationResult(BaseModel):
    """Result schema for cell type annotation."""

    total_cells: int
    annotations: list[CellAnnotation]
    unique_types: list[str]
    output_path: str | None = None


class BatchIntegrationRequest(BaseModel):
    """Request schema for batch integration."""

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
        default=2000,
        ge=500,
        le=5000,
        description="Number of highly variable genes to use",
    )
    output_path: str | None = Field(
        default=None,
        description="Path to save integrated dataset",
    )


class BatchIntegrationResult(BaseModel):
    """Result schema for batch integration."""

    integrated_path: str
    n_cells: int
    n_batches: int
    batch_mixing_score: float = Field(ge=0.0, le=1.0)
    silhouette_score: float = Field(ge=-1.0, le=1.0)


class EmbeddingRequest(BaseModel):
    """Request schema for gene embeddings."""

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
    """Single gene embedding."""

    gene: str
    embedding: list[float]


class EmbeddingResult(BaseModel):
    """Result schema for gene embeddings."""

    embeddings: list[GeneEmbedding]
    embedding_dim: int
    model_used: str
    similarity_matrix: list[list[float]] | None = None
