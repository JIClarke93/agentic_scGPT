"""scGPT model loading and management utilities.

This module provides the ScGPTLoader class for loading, caching, and managing
scGPT model checkpoints and their associated vocabularies.

Example:
    >>> from src.agent.scgpt.services import get_loader
    >>> loader = get_loader()
    >>> model = loader.load_model("scGPT_human")
    >>> vocab = loader.load_vocab("scGPT_human")
"""

import json
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from ..models import ScGPTModelConfig

# Default checkpoint directory (relative to project root)
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"


class ScGPTLoader:
    """Manages loading and caching of scGPT models.

    This class handles model loading, vocabulary management, and device
    selection. Models are cached after first load for efficiency.

    Attributes:
        checkpoint_dir: Directory containing model checkpoints.
        device: The torch device being used (cuda, mps, or cpu).

    Example:
        >>> loader = ScGPTLoader()
        >>> model = loader.load_model("scGPT_human")
        >>> embeddings, genes = loader.get_gene_embeddings_from_model(["TP53"])
    """

    def __init__(self, checkpoint_dir: Path | None = None):
        """Initialize the loader.

        Args:
            checkpoint_dir: Custom directory for model checkpoints.
                Defaults to the package's checkpoints directory.
        """
        self.checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
        self._models: dict[str, Any] = {}
        self._vocabs: dict[str, dict] = {}
        self._device = self._get_device()

    def _get_device(self) -> torch.device:
        """Determine the best available compute device.

        Returns:
            torch.device for CUDA, MPS (Apple Silicon), or CPU.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple MPS device")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        return device

    @property
    def device(self) -> torch.device:
        """Get the current compute device.

        Returns:
            torch.device: The device used for model computations.
        """
        return self._device

    def get_checkpoint_path(self, checkpoint_name: str) -> Path:
        """Get the path to a checkpoint directory.

        Args:
            checkpoint_name: Name of the checkpoint (e.g., "scGPT_human").

        Returns:
            Path to the checkpoint directory.

        Raises:
            FileNotFoundError: If the checkpoint directory does not exist.
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Run: python -m scripts.download_checkpoints {checkpoint_name}"
            )
        return checkpoint_path

    def load_vocab(self, checkpoint_name: str) -> dict[str, int]:
        """Load the gene vocabulary for a checkpoint.

        Vocabularies are cached after first load for efficiency.

        Args:
            checkpoint_name: Name of the checkpoint (e.g., "scGPT_human").

        Returns:
            Dictionary mapping gene symbols to token indices.

        Raises:
            FileNotFoundError: If the checkpoint or vocabulary file is not found.
        """
        if checkpoint_name in self._vocabs:
            return self._vocabs[checkpoint_name]

        checkpoint_path = self.get_checkpoint_path(checkpoint_name)
        vocab_path = checkpoint_path / "vocab.json"

        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")

        with open(vocab_path) as f:
            vocab = json.load(f)

        self._vocabs[checkpoint_name] = vocab
        logger.info(f"Loaded vocabulary with {len(vocab)} genes from {checkpoint_name}")
        return vocab

    def load_model(self, checkpoint_name: str) -> Any:
        """Load a scGPT model from checkpoint.

        The model is cached after first load for efficiency. Subsequent calls
        with the same checkpoint name return the cached model instance.

        Args:
            checkpoint_name: Name of the checkpoint (e.g., "scGPT_human").

        Returns:
            Loaded TransformerModel in evaluation mode on the appropriate device.

        Raises:
            FileNotFoundError: If the checkpoint or model file is not found.
            ImportError: If the scGPT package is not properly installed.
        """
        if checkpoint_name in self._models:
            return self._models[checkpoint_name]

        checkpoint_path = self.get_checkpoint_path(checkpoint_name)
        model_path = checkpoint_path / "best_model.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading scGPT model from {checkpoint_name}...")

        # Load the model state dict
        state_dict = torch.load(model_path, map_location=self._device, weights_only=False)

        # Import scGPT model architecture
        try:
            from scgpt.model import TransformerModel
        except ImportError as e:
            raise ImportError(
                "scGPT not properly installed. "
                "Ensure you have run: uv sync"
            ) from e

        # Get model configuration from state dict or use defaults
        vocab = self.load_vocab(checkpoint_name)
        n_tokens = len(vocab)

        # Standard scGPT configuration
        model_config = ScGPTModelConfig(ntoken=n_tokens, vocab=vocab)

        # Handle different state dict formats
        if "model_state_dict" in state_dict:
            model_state = state_dict["model_state_dict"]
            if "config" in state_dict:
                # Override defaults with saved config
                model_config = ScGPTModelConfig(
                    ntoken=n_tokens,
                    vocab=vocab,
                    **{k: v for k, v in state_dict["config"].items() if k not in ("ntoken", "vocab")},
                )
        else:
            model_state = state_dict

        # Initialize model using config
        config_dict = model_config.model_dump()
        model = TransformerModel(
            ntoken=config_dict["ntoken"],
            d_model=config_dict["d_model"],
            nhead=config_dict["nhead"],
            d_hid=config_dict["d_hid"],
            nlayers=config_dict["nlayers"],
            dropout=config_dict["dropout"],
            pad_token=config_dict["pad_token"],
            vocab=config_dict["vocab"],
        )

        # Load weights
        model.load_state_dict(model_state, strict=False)
        model = model.to(self._device)
        model.eval()

        self._models[checkpoint_name] = model
        logger.info(f"Successfully loaded model {checkpoint_name}")

        return model

    def get_gene_embeddings_from_model(
        self,
        genes: list[str],
        checkpoint_name: str = "scGPT_human",
    ) -> tuple[torch.Tensor, list[str]]:
        """
        Extract gene embeddings directly from the model's embedding layer.

        Args:
            genes: List of gene symbols
            checkpoint_name: Model checkpoint to use

        Returns:
            Tuple of (embeddings tensor, list of found genes)
        """
        model = self.load_model(checkpoint_name)
        vocab = self.load_vocab(checkpoint_name)

        # Get gene indices from vocabulary
        gene_indices = []
        found_genes = []
        missing_genes = []

        for gene in genes:
            if gene in vocab:
                gene_indices.append(vocab[gene])
                found_genes.append(gene)
            else:
                missing_genes.append(gene)

        if missing_genes:
            logger.warning(f"Genes not found in vocabulary: {missing_genes[:10]}...")

        if not gene_indices:
            raise ValueError("No valid genes found in vocabulary")

        # Extract embeddings from the model's embedding layer
        with torch.no_grad():
            indices = torch.tensor(gene_indices, device=self._device)
            embeddings = model.encoder(indices)

        return embeddings, found_genes

    def unload_model(self, checkpoint_name: str) -> None:
        """Unload a model to free memory.

        Removes the model from the cache and clears CUDA memory if applicable.

        Args:
            checkpoint_name: Name of the checkpoint to unload.
        """
        if checkpoint_name in self._models:
            del self._models[checkpoint_name]
            torch.cuda.empty_cache()
            logger.info(f"Unloaded model {checkpoint_name}")

    def unload_all(self) -> None:
        """Unload all cached models to free memory.

        Clears all models from the cache and releases CUDA memory.
        """
        self._models.clear()
        torch.cuda.empty_cache()
        logger.info("Unloaded all models")


# Global singleton instance
_loader: ScGPTLoader | None = None


def get_loader() -> ScGPTLoader:
    """Get the global scGPT loader instance.

    Returns a singleton ScGPTLoader instance, creating it if necessary.
    This ensures models and vocabularies are cached globally.

    Returns:
        The global ScGPTLoader singleton instance.
    """
    global _loader
    if _loader is None:
        _loader = ScGPTLoader()
    return _loader
