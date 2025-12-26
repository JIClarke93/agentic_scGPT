"""Batch integration tool using scGPT."""

import logging
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from scipy.sparse import issparse
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from ..models import BatchIntegrationResult
from ..services import get_loader

logger = logging.getLogger(__name__)


def _preprocess_for_integration(
    adata: sc.AnnData,
    n_hvg: int = 2000,
) -> sc.AnnData:
    """Preprocess AnnData for integration."""
    adata = adata.copy()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=False)

    return adata


def _compute_batch_mixing_score(
    embeddings: np.ndarray,
    batch_labels: np.ndarray,
    n_neighbors: int = 50,
) -> float:
    """
    Compute batch mixing score using k-NN approach.

    Higher score indicates better mixing (0-1 scale).
    """
    if len(np.unique(batch_labels)) < 2:
        return 1.0

    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(embeddings) - 1))
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    mixing_scores = []
    for i, neighbors in enumerate(indices):
        own_batch = batch_labels[i]
        different_batch = np.sum(batch_labels[neighbors] != own_batch)
        mixing_scores.append(different_batch / len(neighbors))

    return float(np.mean(mixing_scores))


def _get_expression_matrix(adata: sc.AnnData) -> np.ndarray:
    """Extract expression matrix as dense numpy array."""
    X = adata.X
    if issparse(X):
        X = X.toarray()
    return np.array(X)


async def integrate_batches(
    dataset_paths: list[str],
    batch_key: str = "batch",
    n_hvg: int = 2000,
    output_path: str | None = None,
) -> BatchIntegrationResult:
    """
    Integrate multiple scRNA-seq datasets with batch correction using scGPT.

    This tool uses scGPT embeddings to create a shared representation space
    where batch effects are minimized while biological variation is preserved.

    Args:
        dataset_paths: List of paths to h5ad files to integrate
        batch_key: Key in adata.obs identifying batch
        n_hvg: Number of highly variable genes to use
        output_path: Optional path to save integrated dataset

    Returns:
        BatchIntegrationResult with integration metrics
    """
    for path in dataset_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

    if len(dataset_paths) < 2:
        raise ValueError("At least 2 datasets required for integration")

    logger.info(f"Integrating {len(dataset_paths)} datasets")

    adatas = []
    for i, path in enumerate(dataset_paths):
        logger.info(f"Loading dataset {i+1}: {path}")
        adata = sc.read_h5ad(path)
        adata.obs[batch_key] = f"batch_{i}"
        adatas.append(adata)
        logger.info(f"  - {adata.n_obs} cells x {adata.n_vars} genes")

    logger.info("Concatenating datasets...")
    adata_combined = sc.concat(adatas, label=batch_key, keys=[f"batch_{i}" for i in range(len(adatas))])
    logger.info(f"Combined: {adata_combined.n_obs} cells x {adata_combined.n_vars} genes")

    logger.info(f"Preprocessing with {n_hvg} highly variable genes...")
    adata_processed = _preprocess_for_integration(adata_combined, n_hvg=n_hvg)

    loader = get_loader()

    try:
        vocab = loader.load_vocab("scGPT_human")
        hvg_mask = adata_processed.var["highly_variable"]
        gene_names = adata_processed.var_names[hvg_mask].tolist()

        valid_genes = [g for g in gene_names if g in vocab]
        logger.info(f"Found {len(valid_genes)}/{len(gene_names)} HVGs in scGPT vocabulary")

        if len(valid_genes) < 100:
            logger.warning("Very few genes match vocabulary. Integration may be suboptimal.")

        model = loader.load_model("scGPT_human")

        gene_ids = torch.tensor([vocab[g] for g in valid_genes], device=loader.device)

        with torch.no_grad():
            gene_embeds = model.encoder(gene_ids)

        gene_mask = adata_processed.var_names.isin(valid_genes)
        adata_subset = adata_processed[:, gene_mask]
        expr_matrix = _get_expression_matrix(adata_subset)

        logger.info("Computing cell embeddings...")
        batch_size = 256
        n_cells = expr_matrix.shape[0]
        cell_embeddings = []

        for batch_start in range(0, n_cells, batch_size):
            batch_end = min(batch_start + batch_size, n_cells)
            batch_expr = expr_matrix[batch_start:batch_end]

            expr_tensor = torch.tensor(batch_expr, dtype=torch.float32, device=loader.device)

            with torch.no_grad():
                cell_embeds = torch.matmul(expr_tensor, gene_embeds)
                cell_embeds = cell_embeds / (expr_tensor.sum(dim=1, keepdim=True) + 1e-8)

            cell_embeddings.append(cell_embeds.cpu().numpy())

            if batch_end % 1000 == 0 or batch_end == n_cells:
                logger.info(f"  Processed {batch_end}/{n_cells} cells")

        all_embeddings = np.vstack(cell_embeddings)
        logger.info(f"Generated embeddings: {all_embeddings.shape}")

        adata_combined.obsm["X_scgpt"] = all_embeddings

        batch_labels = adata_combined.obs[batch_key].values
        batch_codes = np.array([int(b.split("_")[1]) for b in batch_labels])

        logger.info("Computing batch mixing score...")
        batch_mixing = _compute_batch_mixing_score(all_embeddings, batch_codes)

        logger.info("Computing silhouette score...")
        if len(np.unique(batch_codes)) > 1:
            n_sample = min(5000, len(all_embeddings))
            sample_idx = np.random.choice(len(all_embeddings), n_sample, replace=False)
            sil_score = silhouette_score(all_embeddings[sample_idx], batch_codes[sample_idx])
            integration_sil = float(1 - (sil_score + 1) / 2)
        else:
            integration_sil = 1.0

        logger.info("Computing UMAP embedding...")
        sc.pp.neighbors(adata_combined, use_rep="X_scgpt")
        sc.tl.umap(adata_combined)

        if output_path is None:
            output_path = str(Path(dataset_paths[0]).parent / "integrated.h5ad")

        logger.info(f"Saving integrated dataset to {output_path}")
        adata_combined.write_h5ad(output_path)

        return BatchIntegrationResult(
            integrated_path=output_path,
            n_cells=adata_combined.n_obs,
            n_batches=len(dataset_paths),
            batch_mixing_score=round(batch_mixing, 4),
            silhouette_score=round(integration_sil, 4),
        )

    except FileNotFoundError as e:
        logger.error(f"Model checkpoint not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during integration: {e}")
        raise
