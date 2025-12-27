"""FastMCP server exposing scGPT tools.

This module provides an MCP (Model Context Protocol) server that exposes
scGPT single-cell analysis tools to AI agents. The server runs on HTTP
and provides tools for cell annotation, batch integration, and gene embeddings.

Example:
    Run the server from command line::

        $ python -m src.infra.mcp.server

    Or programmatically::

        >>> from src.infra.mcp.server import main
        >>> main()
"""

from fastmcp import FastMCP

from src.agent.scgpt.constants import DEFAULT_BATCH_SIZE, DEFAULT_MCP_PORT, DEFAULT_N_HVG
from src.agent.scgpt.models import AnnotationRequest, BatchIntegrationRequest, EmbeddingRequest
from src.agent.scgpt.tools import annotate_cell_types, get_gene_embeddings, integrate_batches

# Create the MCP server
mcp = FastMCP(
    "scGPT Tools",
    instructions="Single-cell genomics analysis tools powered by scGPT foundation model",
)


@mcp.tool
async def annotate_cells(
    expression_data: str,
    reference_dataset: str = "cellxgene",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """
    Annotate cell types from single-cell RNA-seq data using scGPT.

    Uses the scGPT foundation model to predict cell types based on
    gene expression profiles. Maps cells to known types from a
    reference dataset.

    Args:
        expression_data: Path to h5ad file containing expression data
        reference_dataset: Reference dataset (cellxgene, pbmc, etc.)
        batch_size: Batch size for GPU inference (default: 64)

    Returns:
        Dictionary containing:
        - total_cells: Number of cells annotated
        - annotations: List of cell annotations with confidence scores
        - unique_types: List of unique cell types found
        - output_path: Path to annotated output file
    """
    request = AnnotationRequest(
        expression_data=expression_data,
        reference_dataset=reference_dataset,
        batch_size=batch_size,
    )
    result = await annotate_cell_types(request)
    return result.model_dump()


@mcp.tool
async def integrate_datasets(
    dataset_paths: list[str],
    batch_key: str = "batch",
    n_hvg: int = DEFAULT_N_HVG,
    output_path: str | None = None,
) -> dict:
    """
    Integrate multiple scRNA-seq datasets with batch correction.

    Uses scGPT embeddings to integrate datasets while correcting for
    technical batch effects. Preserves biological variation.

    Args:
        dataset_paths: List of paths to h5ad files to integrate
        batch_key: Key in adata.obs identifying batch (default: "batch")
        n_hvg: Number of highly variable genes (default: 2000)
        output_path: Optional path to save integrated dataset

    Returns:
        Dictionary containing:
        - integrated_path: Path to integrated dataset
        - n_cells: Total number of cells
        - n_batches: Number of batches integrated
        - batch_mixing_score: Quality metric for batch mixing (0-1)
        - silhouette_score: Clustering quality metric (-1 to 1)
    """
    request = BatchIntegrationRequest(
        dataset_paths=dataset_paths,
        batch_key=batch_key,
        n_hvg=n_hvg,
        output_path=output_path,
    )
    result = await integrate_batches(request)
    return result.model_dump()


@mcp.tool
async def extract_gene_embeddings(
    gene_list: list[str],
    model_checkpoint: str = "scGPT_human",
    include_similarity: bool = False,
) -> dict:
    """
    Extract gene embeddings from scGPT for downstream analysis.

    Retrieves learned gene representations that capture functional
    relationships. Useful for gene similarity analysis, clustering,
    or network inference.

    Args:
        gene_list: List of gene symbols to embed (e.g., ["TP53", "BRCA1", "MYC"])
        model_checkpoint: Model to use (scGPT_human, scGPT_mouse)
        include_similarity: If True, compute pairwise similarity matrix

    Returns:
        Dictionary containing:
        - embeddings: List of gene embeddings
        - embedding_dim: Dimension of embedding vectors
        - model_used: Model checkpoint used
        - similarity_matrix: Optional pairwise similarity scores
    """
    request = EmbeddingRequest(
        gene_list=gene_list,
        model_checkpoint=model_checkpoint,
        include_similarity=include_similarity,
    )
    result = await get_gene_embeddings(request)
    return result.model_dump()


def main():
    """Run the MCP server."""
    mcp.run(transport="streamable-http", port=DEFAULT_MCP_PORT)


if __name__ == "__main__":
    main()
