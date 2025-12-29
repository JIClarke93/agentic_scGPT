"""Gene embedding extraction tool using scGPT.

This module provides functionality to extract gene embeddings from the scGPT
foundation model, which can be used for gene similarity analysis, clustering,
or gene network inference.

Example:
    >>> from src.agent.scgpt.tools.embeddings import get_gene_embeddings
    >>> from src.agent.scgpt.models import EmbeddingRequest
    >>> request = EmbeddingRequest(
    ...     gene_list=["TP53", "BRCA1", "MYC"],
    ...     include_similarity=True
    ... )
    >>> result = await get_gene_embeddings(request)
    >>> print(f"Got {len(result.embeddings)} embeddings of dim {result.embedding_dim}")
"""

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from temporalio import activity

from ..models import EmbeddingRequest, EmbeddingResponse, GeneEmbedding
from ..services import get_loader


@activity.defn
async def get_gene_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Extract gene embeddings from scGPT for downstream analysis.

    Retrieves the learned gene representations from the scGPT model's
    embedding layer, optionally computing pairwise similarity scores.

    Args:
        request: EmbeddingRequest containing gene list, model checkpoint,
            and similarity computation options.

    Returns:
        EmbeddingResponse with gene embeddings and optional similarity matrix.

    Raises:
        ValueError: If gene_list is empty.
        FileNotFoundError: If model checkpoint is not found.
    """
    if not request.gene_list:
        raise ValueError("gene_list cannot be empty")

    logger.info(f"Extracting embeddings for {len(request.gene_list)} genes")
    logger.info(f"Using model checkpoint: {request.model_checkpoint}")

    loader = get_loader()

    try:
        embeddings_tensor, found_genes = loader.get_gene_embeddings_from_model(
            genes=request.gene_list,
            checkpoint_name=request.model_checkpoint,
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
        if request.include_similarity and len(embeddings) > 0:
            logger.info("Computing pairwise cosine similarity matrix")
            sim_matrix = cosine_similarity(embeddings_np)
            similarity_matrix = sim_matrix.tolist()

        missing = set(request.gene_list) - set(found_genes)
        if missing:
            logger.warning(
                f"{len(missing)} genes not found in vocabulary: "
                f"{list(missing)[:5]}{'...' if len(missing) > 5 else ''}"
            )

        return EmbeddingResponse(
            embeddings=embeddings,
            embedding_dim=embedding_dim,
            model_used=request.model_checkpoint,
            similarity_matrix=similarity_matrix,
        )

    except FileNotFoundError as e:
        logger.error(f"Model checkpoint not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error extracting embeddings: {e}")
        raise
