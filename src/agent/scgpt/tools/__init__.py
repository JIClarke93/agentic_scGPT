"""scGPT tools for single-cell analysis.

This package provides the core analysis tools for working with single-cell
RNA-seq data using the scGPT foundation model.

Available tools:
    annotate_cell_types: Predict cell types from expression data.
    get_gene_embeddings: Extract gene embeddings for similarity analysis.
    integrate_batches: Integrate multiple datasets with batch correction.
"""

from .annotate import annotate_cell_types
from .embeddings import get_gene_embeddings
from .integrate import integrate_batches

__all__ = [
    "annotate_cell_types",
    "get_gene_embeddings",
    "integrate_batches",
]
