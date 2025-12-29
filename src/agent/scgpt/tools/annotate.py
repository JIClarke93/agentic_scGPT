"""Cell type annotation tool using scGPT.

This module provides functionality to annotate cell types from single-cell
RNA-seq data using the scGPT foundation model embeddings and marker gene
expression patterns.

Example:
    >>> from src.agent.scgpt.tools.annotate import annotate_cell_types
    >>> from src.agent.scgpt.models import AnnotationRequest
    >>> request = AnnotationRequest(expression_data="data/pbmc.h5ad")
    >>> result = await annotate_cell_types(request)
    >>> print(f"Found {len(result.unique_types)} cell types")
"""

from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from loguru import logger
from scipy.sparse import issparse
from temporalio import activity

from ..models import (
    AlternativeCellType,
    AnnotationRequest,
    AnnotationResponse,
    CellAnnotation,
    CellTypePredictionRequest,
    CellTypePredictionResponse,
)
from ..services import get_loader
from ..constants import (
    CONFIDENCE_MULTIPLIER,
    MARKER_GENES,
    MAX_CONFIDENCE,
    MIN_ALTERNATIVE_PROB,
    MIN_CELLS_PER_GENE,
    MIN_GENES_PER_CELL,
    MIN_VALID_GENES_WARNING,
    N_TOP_ALTERNATIVES,
    NORMALIZATION_TARGET_SUM,
)



def _preprocess_adata(adata: sc.AnnData, n_hvg: int = 2000) -> sc.AnnData:
    """Preprocess AnnData for scGPT embedding.

    Applies standard single-cell preprocessing steps including quality filtering,
    normalization, log transformation, and highly variable gene selection.

    Args:
        adata: AnnData object containing raw expression data.
        n_hvg: Number of highly variable genes to select.

    Returns:
        Preprocessed AnnData object with HVG subset.
    """
    adata = adata.copy()

    # Basic preprocessing
    sc.pp.filter_cells(adata, min_genes=MIN_GENES_PER_CELL)
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)

    # Normalize and log transform
    sc.pp.normalize_total(adata, target_sum=NORMALIZATION_TARGET_SUM)
    sc.pp.log1p(adata)

    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=True)

    return adata


def _get_expression_matrix(adata: sc.AnnData) -> np.ndarray:
    """Extract expression matrix as dense numpy array.

    Args:
        adata: AnnData object containing expression data.

    Returns:
        Dense numpy array of shape (n_cells, n_genes).
    """
    X = adata.X
    if X is None:
        raise ValueError("AnnData object has no expression matrix (X is None)")
    if issparse(X):
        X = X.toarray()  # type: ignore[union-attr]
    return np.array(X)


@activity.defn
async def annotate_cell_types(request: AnnotationRequest) -> AnnotationResponse:
    """Annotate cell types from single-cell RNA-seq data using scGPT embeddings.

    Uses the scGPT foundation model to generate cell embeddings and maps them
    to known cell types based on marker gene expression patterns.

    Args:
        request: AnnotationRequest containing expression data path, reference
            dataset, and batch size configuration.

    Returns:
        AnnotationResponse with predicted cell types and confidence scores.

    Raises:
        FileNotFoundError: If the expression data file or model checkpoint
            is not found.
    """
    # Validate input path
    data_path = Path(request.expression_data)
    if not data_path.exists():
        raise FileNotFoundError(f"Expression data not found: {request.expression_data}")

    logger.info(f"Loading expression data from {request.expression_data}")

    # Load the data
    adata = sc.read_h5ad(request.expression_data)
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

        if len(valid_genes) < MIN_VALID_GENES_WARNING:
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

        for batch_start in range(0, n_cells, request.batch_size):
            batch_end = min(batch_start + request.batch_size, n_cells)
            batch_expr = expr_matrix[batch_start:batch_end]

            # Convert expression to tensor
            expr_tensor = torch.tensor(batch_expr, dtype=torch.float32, device=loader.device)

            # Get cell embeddings using scGPT
            with torch.no_grad():
                gene_embeds = model.encoder(gene_ids)
                cell_embeds = torch.matmul(expr_tensor, gene_embeds)
                cell_embeds = cell_embeds / (expr_tensor.sum(dim=1, keepdim=True) + 1e-8)

            cell_embeds_np = cell_embeds.cpu().numpy()

            # Extract expression matrix for this batch
            X_processed = adata_processed.X
            if X_processed is None:
                raise ValueError("Processed AnnData has no expression matrix")
            batch_expr_raw = X_processed[batch_start:batch_end]  # type: ignore[index]
            if issparse(batch_expr_raw):
                batch_expr_raw = batch_expr_raw.toarray()  # type: ignore[union-attr]
            else:
                batch_expr_raw = np.array(batch_expr_raw)

            gene_names = adata_processed.var_names.tolist()

            for i, (cell_idx, embed) in enumerate(zip(range(batch_start, batch_end), cell_embeds_np)):
                prediction_request = CellTypePredictionRequest(
                    embedding=embed.tolist(),
                    expression=batch_expr_raw[i].tolist(),
                    gene_names=gene_names,
                )
                prediction = _predict_cell_type(prediction_request)

                annotations.append(
                    CellAnnotation(
                        cell_id=cell_ids[cell_idx],
                        predicted_type=prediction.predicted_type,
                        confidence=prediction.confidence,
                        alternative_types=[
                            (alt.cell_type, alt.probability)
                            for alt in prediction.alternatives
                        ],
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

        return AnnotationResponse(
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


def _predict_cell_type(request: CellTypePredictionRequest) -> CellTypePredictionResponse:
    """Predict cell type from embedding using marker gene heuristics.

    Uses a simplified approach based on marker gene expression scores.
    In production, this should be replaced with a trained classifier
    or k-NN against a reference dataset.

    Args:
        request: CellTypePredictionRequest containing the cell embedding,
            expression values, and gene names.

    Returns:
        CellTypePredictionResponse with predicted type, confidence, and alternatives.
    """
    expr = np.array(request.expression)
    gene_to_idx = {g: i for i, g in enumerate(request.gene_names)}

    scores = {}
    for cell_type, markers in MARKER_GENES.items():
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
    confidence = min(sorted_types[0][1] * CONFIDENCE_MULTIPLIER, MAX_CONFIDENCE)

    alternatives = [
        AlternativeCellType(cell_type=t, probability=round(p, 3))
        for t, p in sorted_types[1 : N_TOP_ALTERNATIVES + 1]
        if p > MIN_ALTERNATIVE_PROB
    ]

    return CellTypePredictionResponse(
        predicted_type=predicted_type,
        confidence=round(confidence, 3),
        alternatives=alternatives,
    )
