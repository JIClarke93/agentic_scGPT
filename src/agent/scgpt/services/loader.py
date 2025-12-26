"""scGPT model loading and management utilities."""

import json
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Default checkpoint directory (relative to project root)
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"


class ScGPTLoader:
    """Manages loading and caching of scGPT models."""

    def __init__(self, checkpoint_dir: Path | None = None):
        self.checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
        self._models: dict[str, Any] = {}
        self._vocabs: dict[str, dict] = {}
        self._device = self._get_device()

    def _get_device(self) -> torch.device:
        """Determine the best available device."""
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
        """Get the current device."""
        return self._device

    def get_checkpoint_path(self, checkpoint_name: str) -> Path:
        """Get the path to a checkpoint directory."""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Run: python -m scripts.download_checkpoints {checkpoint_name}"
            )
        return checkpoint_path

    def load_vocab(self, checkpoint_name: str) -> dict[str, int]:
        """Load the gene vocabulary for a checkpoint."""
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
        """
        Load a scGPT model from checkpoint.

        The model is cached after first load for efficiency.
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
        model_config = {
            "ntoken": n_tokens,
            "d_model": 512,
            "nhead": 8,
            "d_hid": 512,
            "nlayers": 12,
            "dropout": 0.2,
            "pad_token": "<pad>",
            "vocab": vocab,
        }

        # Handle different state dict formats
        if "model_state_dict" in state_dict:
            model_state = state_dict["model_state_dict"]
            if "config" in state_dict:
                model_config.update(state_dict["config"])
        else:
            model_state = state_dict

        # Initialize model
        model = TransformerModel(
            ntoken=model_config["ntoken"],
            d_model=model_config["d_model"],
            nhead=model_config["nhead"],
            d_hid=model_config["d_hid"],
            nlayers=model_config["nlayers"],
            dropout=model_config["dropout"],
            pad_token=model_config["pad_token"],
            vocab=model_config["vocab"],
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
        """Unload a model to free memory."""
        if checkpoint_name in self._models:
            del self._models[checkpoint_name]
            torch.cuda.empty_cache()
            logger.info(f"Unloaded model {checkpoint_name}")

    def unload_all(self) -> None:
        """Unload all models."""
        self._models.clear()
        torch.cuda.empty_cache()
        logger.info("Unloaded all models")


# Global singleton instance
_loader: ScGPTLoader | None = None


def get_loader() -> ScGPTLoader:
    """Get the global scGPT loader instance."""
    global _loader
    if _loader is None:
        _loader = ScGPTLoader()
    return _loader
