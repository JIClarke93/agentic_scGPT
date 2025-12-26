"""Cell type annotation tool using scGPT."""

import logging
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from scipy.sparse import issparse

from ..models import AnnotationResult, CellAnnotation
from ..services import get_loader

logger = logging.getLogger(__name__)


def _preprocess_adata(adata: sc.AnnData, n_hvg: int = 2000) -> sc.AnnData:
    """Preprocess AnnData for scGPT embedding."""
    adata = adata.copy()

    # Basic preprocessing
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Normalize and log transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=True)

    return adata


def _get_expression_matrix(adata: sc.AnnData) -> np.ndarray:
    """Extract expression matrix as dense numpy array."""
    X = adata.X
    if issparse(X):
        X = X.toarray()
    return np.array(X)


async def annotate_cell_types(
    expression_data: str,
    reference_dataset: str = "cellxgene",
    batch_size: int = 64,
) -> AnnotationResult:
    """
    Annotate cell types from single-cell RNA-seq data using scGPT embeddings.

    This tool uses the scGPT foundation model to generate cell embeddings
    and maps them to known cell types. Currently uses a simplified approach
    based on marker gene expression until reference datasets are integrated.

    Args:
        expression_data: Path to h5ad file containing expression data
        reference_dataset: Reference dataset for annotation (currently informational)
        batch_size: Batch size for GPU inference

    Returns:
        AnnotationResult with predicted cell types and confidence scores
    """
    # Validate input path
    data_path = Path(expression_data)
    if not data_path.exists():
        raise FileNotFoundError(f"Expression data not found: {expression_data}")

    logger.info(f"Loading expression data from {expression_data}")

    # Load the data
    adata = sc.read_h5ad(expression_data)
    logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")

    # Store original cell IDs
    if adata.obs.index.name is None:
        cell_ids = [f"cell_{i:05d}" for i in range(adata.n_obs)]
    else:
        cell_ids = adata.obs.index.tolist()

    # Preprocess
    logger.info("Preprocessing data...")
    adata_processed = _preprocess_adata(adata)
    logger.info(f"After preprocessing: {adata_processed.n_obs} cells x {adata_processed.n_vars} genes")

    # Get the scGPT loader
    loader = get_loader()

    try:
        # Load vocabulary to map genes
        vocab = loader.load_vocab("scGPT_human")
        gene_names = adata_processed.var_names.tolist()

        # Find genes present in vocabulary
        valid_genes = [g for g in gene_names if g in vocab]
        logger.info(f"Found {len(valid_genes)}/{len(gene_names)} genes in scGPT vocabulary")

        if len(valid_genes) < 100:
            logger.warning("Very few genes match vocabulary. Results may be unreliable.")

        # Get model for cell embedding
        model = loader.load_model("scGPT_human")

        # Prepare gene tokens
        gene_ids = torch.tensor([vocab[g] for g in valid_genes], device=loader.device)

        # Get expression values for valid genes
        gene_mask = adata_processed.var_names.isin(valid_genes)
        expr_matrix = _get_expression_matrix(adata_processed[:, gene_mask])

        # Process cells in batches
        annotations = []
        n_cells = adata_processed.n_obs

        for batch_start in range(0, n_cells, batch_size):
            batch_end = min(batch_start + batch_size, n_cells)
            batch_expr = expr_matrix[batch_start:batch_end]

            # Convert expression to tensor
            expr_tensor = torch.tensor(batch_expr, dtype=torch.float32, device=loader.device)

            # Get cell embeddings using scGPT
            with torch.no_grad():
                gene_embeds = model.encoder(gene_ids)
                cell_embeds = torch.matmul(expr_tensor, gene_embeds)
                cell_embeds = cell_embeds / (expr_tensor.sum(dim=1, keepdim=True) + 1e-8)

            cell_embeds_np = cell_embeds.cpu().numpy()

            for i, (cell_idx, embed) in enumerate(zip(range(batch_start, batch_end), cell_embeds_np)):
                predicted_type, confidence, alternatives = _predict_cell_type(
                    embed, adata_processed, cell_idx, valid_genes
                )

                annotations.append(
                    CellAnnotation(
                        cell_id=cell_ids[cell_idx],
                        predicted_type=predicted_type,
                        confidence=confidence,
                        alternative_types=alternatives,
                    )
                )

            logger.info(f"Processed {batch_end}/{n_cells} cells")

        unique_types = sorted(set(a.predicted_type for a in annotations))

        # Save annotated data
        output_path = data_path.with_suffix(".annotated.h5ad")
        adata.obs["predicted_cell_type"] = [a.predicted_type for a in annotations]
        adata.obs["prediction_confidence"] = [a.confidence for a in annotations]
        adata.write_h5ad(output_path)
        logger.info(f"Saved annotated data to {output_path}")

        return AnnotationResult(
            total_cells=len(annotations),
            annotations=annotations,
            unique_types=unique_types,
            output_path=str(output_path),
        )

    except FileNotFoundError as e:
        logger.error(f"Model checkpoint not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during annotation: {e}")
        raise


def _predict_cell_type(
    embedding: np.ndarray,
    adata: sc.AnnData,
    cell_idx: int,
    valid_genes: list[str],
) -> tuple[str, float, list[tuple[str, float]]]:
    """
    Predict cell type from embedding using marker gene heuristics.

    This is a simplified approach. In production, use a trained classifier
    or k-NN against a reference dataset.
    """
    marker_genes = {
        "T cell": ["CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B"],
        "B cell": ["CD19", "MS4A1", "CD79A", "CD79B"],
        "NK cell": ["NKG7", "GNLY", "KLRD1", "NCAM1"],
        "Monocyte": ["CD14", "LYZ", "FCGR3A", "MS4A7"],
        "Macrophage": ["CD68", "CD163", "MRC1", "MARCO"],
        "Dendritic cell": ["CD1C", "FCER1A", "CLEC10A", "CD141"],
        "Neutrophil": ["FCGR3B", "CSF3R", "S100A8", "S100A9"],
        "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5"],
        "Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM"],
        "Epithelial": ["EPCAM", "KRT18", "KRT19", "CDH1"],
    }

    expr = adata.X[cell_idx]
    if issparse(expr):
        expr = expr.toarray().flatten()
    else:
        expr = np.array(expr).flatten()

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

    scores = {}
    for cell_type, markers in marker_genes.items():
        marker_expr = []
        for marker in markers:
            if marker in gene_to_idx:
                marker_expr.append(expr[gene_to_idx[marker]])
        if marker_expr:
            scores[cell_type] = np.mean(marker_expr)
        else:
            scores[cell_type] = 0.0

    total = sum(scores.values()) + 1e-8
    probs = {k: v / total for k, v in scores.items()}

    sorted_types = sorted(probs.items(), key=lambda x: -x[1])

    predicted_type = sorted_types[0][0]
    confidence = min(sorted_types[0][1] * 2, 0.99)

    alternatives = [(t, round(p, 3)) for t, p in sorted_types[1:4] if p > 0.01]

    return predicted_type, round(confidence, 3), alternatives
