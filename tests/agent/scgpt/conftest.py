"""Pytest configuration and shared fixtures."""

# =============================================================================
# Test Constants
# =============================================================================

# Fixture dimensions
TEST_N_CELLS = 100
TEST_N_GENES = 500
TEST_EMBEDDING_DIM = 512

# Data generation
TEST_RANDOM_SEED = 42
TEST_POISSON_LAMBDA = 2

# Batch sizes for tests
TEST_BATCH_SIZE = 32
TEST_N_HVG = 500
TEST_N_NEIGHBORS = 10
TEST_N_SAMPLES = 50

# Marker genes used in fixtures
TEST_MARKER_GENES = ["CD3D", "CD3E", "CD19", "MS4A1", "CD14", "LYZ", "NKG7"]
TEST_STANDARD_GENES = [
    "CD3D", "CD3E", "CD19", "MS4A1", "CD14",
    "LYZ", "NKG7", "CD68", "COL1A1", "EPCAM",
]

# Gene lists for embedding tests
TEST_EMBEDDING_GENES = ["TP53", "BRCA1", "MYC"]
