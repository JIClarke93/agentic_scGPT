"""Gene embedding extraction tool using scGPT."""

import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..models import EmbeddingResult, GeneEmbedding
from ..services import get_loader

logger = logging.getLogger(__name__)


async def get_gene_embeddings(
    gene_list: list[str],
    model_checkpoint: str = "scGPT_human",
    include_similarity: bool = False,
) -> EmbeddingResult:
    """
    Extract gene embeddings from scGPT for downstream analysis.

    This tool retrieves the learned gene representations from scGPT,
    which can be used for gene similarity analysis, clustering, or
    gene network inference.

    Args:
        gene_list: List of gene symbols to embed
        model_checkpoint: Model checkpoint to use (scGPT_human, scGPT_CP, scGPT_BC)
        include_similarity: If True, compute pairwise cosine similarity matrix

    Returns:
        EmbeddingResult with gene embeddings and optional similarity matrix
    """
    if not gene_list:
        raise ValueError("gene_list cannot be empty")

    logger.info(f"Extracting embeddings for {len(gene_list)} genes")
    logger.info(f"Using model checkpoint: {model_checkpoint}")

    loader = get_loader()

    try:
        embeddings_tensor, found_genes = loader.get_gene_embeddings_from_model(
            genes=gene_list,
            checkpoint_name=model_checkpoint,
        )

        embeddings_np = embeddings_tensor.cpu().numpy()
        embedding_dim = embeddings_np.shape[1]

        embeddings = []
        for gene, embedding in zip(found_genes, embeddings_np):
            embeddings.append(
                GeneEmbedding(
                    gene=gene,
                    embedding=embedding.tolist(),
                )
            )

        similarity_matrix = None
        if include_similarity and len(embeddings) > 0:
            logger.info("Computing pairwise cosine similarity matrix")
            sim_matrix = cosine_similarity(embeddings_np)
            similarity_matrix = sim_matrix.tolist()

        missing = set(gene_list) - set(found_genes)
        if missing:
            logger.warning(
                f"{len(missing)} genes not found in vocabulary: "
                f"{list(missing)[:5]}{'...' if len(missing) > 5 else ''}"
            )

        return EmbeddingResult(
            embeddings=embeddings,
            embedding_dim=embedding_dim,
            model_used=model_checkpoint,
            similarity_matrix=similarity_matrix,
        )

    except FileNotFoundError as e:
        logger.error(f"Model checkpoint not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error extracting embeddings: {e}")
        raise
