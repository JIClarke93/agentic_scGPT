"""Constants for scGPT agent and tools.

This module defines all configuration constants used throughout the scGPT
agent and tools, including server configuration, batch processing parameters,
gene filtering thresholds, model architecture defaults, and cell type markers.

Attributes:
    DEFAULT_MCP_SERVER_URL: Default URL for the MCP server endpoint.
    DEFAULT_MCP_PORT: Default port for the MCP server.
    DEFAULT_MODEL: Default LLM model identifier for the agent.
    DEFAULT_BATCH_SIZE: Default batch size for GPU inference.
    DEFAULT_N_HVG: Default number of highly variable genes to select.
    MARKER_GENES: Mapping of cell types to their marker gene lists.
    COMMON_MARKERS: Subset of commonly used marker genes.
"""

# =============================================================================
# Server and Agent Configuration
# =============================================================================
DEFAULT_MCP_SERVER_URL: str = "http://localhost:8000/mcp"
DEFAULT_MCP_PORT: int = 8000
DEFAULT_MODEL: str = "anthropic:claude-sonnet-4-20250514"

# =============================================================================
# Batch Processing
# =============================================================================
DEFAULT_BATCH_SIZE: int = 64
MIN_BATCH_SIZE: int = 1
MAX_BATCH_SIZE: int = 512
INTEGRATION_BATCH_SIZE: int = 256

# =============================================================================
# Gene Filtering
# =============================================================================
DEFAULT_N_HVG: int = 2000
MIN_N_HVG: int = 500
MAX_N_HVG: int = 5000
MIN_GENES_PER_CELL: int = 200
MIN_CELLS_PER_GENE: int = 3
MIN_VALID_GENES_WARNING: int = 100

# =============================================================================
# Cell Type Prediction
# =============================================================================
CONFIDENCE_MULTIPLIER: float = 2.0
MAX_CONFIDENCE: float = 0.99
MIN_ALTERNATIVE_PROB: float = 0.01
N_TOP_ALTERNATIVES: int = 3

# =============================================================================
# Integration Metrics
# =============================================================================
DEFAULT_N_NEIGHBORS: int = 50
MAX_SILHOUETTE_SAMPLE_SIZE: int = 5000
LOG_PROGRESS_INTERVAL: int = 1000

# =============================================================================
# Data Preprocessing
# =============================================================================
NORMALIZATION_TARGET_SUM: float = 1e4

# =============================================================================
# Model Architecture (scGPT defaults)
# =============================================================================
MODEL_D_MODEL: int = 512
MODEL_NHEAD: int = 8
MODEL_D_HID: int = 512
MODEL_N_LAYERS: int = 12
MODEL_DROPOUT: float = 0.2

# =============================================================================
# Cell Type Marker Genes
# =============================================================================
MARKER_GENES: dict[str, list[str]] = {
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

# Common marker genes for quick reference (subset of above)
COMMON_MARKERS: list[str] = ["CD3D", "CD4", "CD8A", "MS4A1", "CD14"]
