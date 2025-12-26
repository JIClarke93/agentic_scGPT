"""Tests for scGPT tools with snapshot testing."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from syrupy.assertion import SnapshotAssertion

from src.agent.scgpt.models import AnnotationResult, BatchIntegrationResult, EmbeddingResult


@pytest.fixture
def mock_loader():
    """Create a mock scGPT loader."""
    mock = MagicMock()
    mock.device = "cpu"

    # Mock vocabulary
    mock.load_vocab.return_value = {
        "TP53": 0,
        "BRCA1": 1,
        "MYC": 2,
        "CD3D": 3,
        "CD3E": 4,
        "CD19": 5,
        "MS4A1": 6,
        "CD14": 7,
        "LYZ": 8,
        "NKG7": 9,
    }

    # Mock model with encoder
    mock_model = MagicMock()

    def mock_encoder(indices):
        # Return fake embeddings of shape (n_genes, 512)
        n = len(indices) if hasattr(indices, "__len__") else indices.shape[0]
        return torch.randn(n, 512)

    mock_model.encoder = mock_encoder
    mock_model.eval.return_value = mock_model
    mock.load_model.return_value = mock_model

    # Mock get_gene_embeddings_from_model
    def mock_get_embeddings(genes, checkpoint_name):
        found = [g for g in genes if g in mock.load_vocab()]
        embeddings = torch.randn(len(found), 512)
        return embeddings, found

    mock.get_gene_embeddings_from_model = mock_get_embeddings

    return mock


@pytest.fixture
def deterministic_mock_loader():
    """Create a deterministic mock scGPT loader for snapshot testing."""
    mock = MagicMock()
    mock.device = "cpu"

    # Mock vocabulary
    vocab = {
        "TP53": 0,
        "BRCA1": 1,
        "MYC": 2,
        "EGFR": 3,
        "CD3D": 4,
        "CD3E": 5,
        "CD19": 6,
        "MS4A1": 7,
        "CD14": 8,
        "LYZ": 9,
    }
    mock.load_vocab.return_value = vocab

    # Mock model with deterministic encoder
    mock_model = MagicMock()

    def mock_encoder(indices):
        # Return deterministic embeddings based on index
        torch.manual_seed(42)
        n = len(indices) if hasattr(indices, "__len__") else indices.shape[0]
        return torch.randn(n, 512)

    mock_model.encoder = mock_encoder
    mock_model.eval.return_value = mock_model
    mock.load_model.return_value = mock_model

    # Mock get_gene_embeddings_from_model with deterministic output
    def mock_get_embeddings(genes, checkpoint_name):
        torch.manual_seed(42)
        found = [g for g in genes if g in vocab]
        # Create deterministic embeddings - each gene gets a unique but consistent embedding
        embeddings = []
        for i, gene in enumerate(found):
            torch.manual_seed(42 + vocab[gene])
            embeddings.append(torch.randn(512))
        if embeddings:
            return torch.stack(embeddings), found
        return torch.empty(0, 512), found

    mock.get_gene_embeddings_from_model = mock_get_embeddings

    return mock


@pytest.fixture
def sample_h5ad(tmp_path):
    """Create a sample h5ad file for testing."""
    # Create random expression data with enough genes per cell
    n_cells = 100
    n_genes = 500  # More genes to survive filtering

    # Use denser matrix to ensure cells have enough genes
    # Real scRNA-seq is sparse but we need cells to have 200+ genes
    np.random.seed(42)
    X = np.random.poisson(2, size=(n_cells, n_genes)).astype(float)
    X = sp.csr_matrix(X)

    # Gene names including some markers
    markers = ["CD3D", "CD3E", "CD19", "MS4A1", "CD14", "LYZ", "NKG7"]
    gene_names = markers + [f"GENE_{i}" for i in range(n_genes - len(markers))]

    # Cell IDs
    cell_ids = [f"cell_{i:05d}" for i in range(n_cells)]

    # Create AnnData
    adata = ad.AnnData(
        X=X,
        obs={"cell_id": cell_ids},
        var={"gene_symbol": gene_names},
    )
    adata.obs.index = cell_ids
    adata.var.index = gene_names

    # Save to file
    filepath = tmp_path / "test_data.h5ad"
    adata.write_h5ad(filepath)

    return str(filepath)


@pytest.fixture
def sample_h5ad_pair(tmp_path):
    """Create a pair of h5ad files for integration testing."""
    paths = []
    np.random.seed(42)

    for batch_idx in range(2):
        n_cells = 100
        n_genes = 500  # More genes to survive filtering

        # Dense enough data to survive preprocessing
        X = np.random.poisson(2, size=(n_cells, n_genes)).astype(float)
        X = sp.csr_matrix(X)

        gene_names = [f"GENE_{i}" for i in range(n_genes)]
        cell_ids = [f"batch{batch_idx}_cell_{i:04d}" for i in range(n_cells)]

        adata = ad.AnnData(X=X)
        adata.obs.index = cell_ids
        adata.var.index = gene_names

        filepath = tmp_path / f"batch_{batch_idx}.h5ad"
        adata.write_h5ad(filepath)
        paths.append(str(filepath))

    return paths


class TestGetGeneEmbeddings:
    """Tests for gene embedding extraction."""

    @pytest.mark.asyncio
    async def test_get_embeddings_basic(self, mock_loader):
        """Test basic embedding extraction."""
        with patch("src.agent.scgpt.tools.embeddings.get_loader", return_value=mock_loader):
            from src.agent.scgpt.tools.embeddings import get_gene_embeddings

            result = await get_gene_embeddings(
                gene_list=["TP53", "BRCA1", "MYC"],
                model_checkpoint="scGPT_human",
                include_similarity=False,
            )

            assert isinstance(result, EmbeddingResult)
            assert result.model_used == "scGPT_human"
            assert result.embedding_dim == 512
            assert result.similarity_matrix is None

    @pytest.mark.asyncio
    async def test_get_embeddings_with_similarity(self, mock_loader):
        """Test embedding extraction with similarity matrix."""
        with patch("src.agent.scgpt.tools.embeddings.get_loader", return_value=mock_loader):
            from src.agent.scgpt.tools.embeddings import get_gene_embeddings

            result = await get_gene_embeddings(
                gene_list=["TP53", "BRCA1"],
                model_checkpoint="scGPT_human",
                include_similarity=True,
            )

            assert result.similarity_matrix is not None
            assert len(result.similarity_matrix) == len(result.embeddings)

    @pytest.mark.asyncio
    async def test_get_embeddings_empty_list(self, mock_loader):
        """Test that empty gene list raises error."""
        with patch("src.agent.scgpt.tools.embeddings.get_loader", return_value=mock_loader):
            from src.agent.scgpt.tools.embeddings import get_gene_embeddings

            with pytest.raises(ValueError, match="cannot be empty"):
                await get_gene_embeddings(gene_list=[])


class TestAnnotateCellTypes:
    """Tests for cell type annotation."""

    @pytest.mark.asyncio
    async def test_annotate_file_not_found(self, mock_loader):
        """Test that missing file raises error."""
        with patch("src.agent.scgpt.tools.annotate.get_loader", return_value=mock_loader):
            from src.agent.scgpt.tools.annotate import annotate_cell_types

            with pytest.raises(FileNotFoundError):
                await annotate_cell_types(
                    expression_data="/nonexistent/path.h5ad"
                )

    @pytest.mark.asyncio
    async def test_annotate_basic(self, mock_loader, sample_h5ad):
        """Test basic cell type annotation."""
        with patch("src.agent.scgpt.tools.annotate.get_loader", return_value=mock_loader):
            from src.agent.scgpt.tools.annotate import annotate_cell_types

            result = await annotate_cell_types(
                expression_data=sample_h5ad,
                batch_size=32,
            )

            assert isinstance(result, AnnotationResult)
            assert result.total_cells > 0
            assert len(result.annotations) == result.total_cells
            assert len(result.unique_types) > 0
            assert result.output_path is not None

            # Check output file was created
            assert Path(result.output_path).exists()


class TestIntegrateBatches:
    """Tests for batch integration."""

    @pytest.mark.asyncio
    async def test_integrate_single_dataset_error(self, mock_loader, sample_h5ad):
        """Test that single dataset raises error."""
        with patch("src.agent.scgpt.tools.integrate.get_loader", return_value=mock_loader):
            from src.agent.scgpt.tools.integrate import integrate_batches

            with pytest.raises(ValueError, match="At least 2 datasets"):
                await integrate_batches(dataset_paths=[sample_h5ad])

    @pytest.mark.asyncio
    async def test_integrate_file_not_found(self, mock_loader):
        """Test that missing file raises error."""
        with patch("src.agent.scgpt.tools.integrate.get_loader", return_value=mock_loader):
            from src.agent.scgpt.tools.integrate import integrate_batches

            with pytest.raises(FileNotFoundError):
                await integrate_batches(
                    dataset_paths=["/nonexistent/a.h5ad", "/nonexistent/b.h5ad"]
                )

    @pytest.mark.asyncio
    async def test_integrate_basic(self, mock_loader, sample_h5ad_pair):
        """Test basic batch integration."""
        with patch("src.agent.scgpt.tools.integrate.get_loader", return_value=mock_loader):
            from src.agent.scgpt.tools.integrate import integrate_batches

            result = await integrate_batches(
                dataset_paths=sample_h5ad_pair,
                n_hvg=500,
            )

            assert isinstance(result, BatchIntegrationResult)
            assert result.n_batches == 2
            assert result.n_cells > 0
            assert 0 <= result.batch_mixing_score <= 1
            assert 0 <= result.silhouette_score <= 1
            assert Path(result.integrated_path).exists()


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_predict_cell_type(self):
        """Test cell type prediction helper."""
        from src.agent.scgpt.tools.annotate import _predict_cell_type

        # Create mock AnnData with T cell markers
        import anndata as ad
        import numpy as np

        # Expression data where CD3D, CD3E are highly expressed
        X = np.zeros((1, 10))
        X[0, 0] = 5.0  # CD3D
        X[0, 1] = 4.0  # CD3E

        adata = ad.AnnData(X=X)
        adata.var.index = [
            "CD3D", "CD3E", "CD19", "MS4A1", "CD14",
            "LYZ", "NKG7", "CD68", "COL1A1", "EPCAM"
        ]

        cell_type, confidence, alternatives = _predict_cell_type(
            embedding=np.zeros(512),  # Not used in current implementation
            adata=adata,
            cell_idx=0,
            valid_genes=list(adata.var.index),
        )

        # Should predict T cell due to CD3D/CD3E expression
        assert cell_type == "T cell"
        assert confidence > 0

    def test_compute_batch_mixing_score(self):
        """Test batch mixing score computation."""
        from src.agent.scgpt.tools.integrate import _compute_batch_mixing_score

        # Create embeddings where batches are well-mixed
        np.random.seed(42)
        embeddings = np.random.randn(100, 10)
        batch_labels = np.array([0, 1] * 50)

        score = _compute_batch_mixing_score(embeddings, batch_labels, n_neighbors=10)

        # Random embeddings should have decent mixing
        assert 0 <= score <= 1

    def test_compute_batch_mixing_single_batch(self):
        """Test batch mixing with single batch returns 1.0."""
        from src.agent.scgpt.tools.integrate import _compute_batch_mixing_score

        embeddings = np.random.randn(100, 10)
        batch_labels = np.zeros(100)

        score = _compute_batch_mixing_score(embeddings, batch_labels)

        assert score == 1.0


class TestSnapshotTools:
    """Snapshot tests for tool outputs."""

    def test_predict_cell_type_t_cell_snapshot(self, snapshot: SnapshotAssertion):
        """Snapshot test for T cell prediction."""
        from src.agent.scgpt.tools.annotate import _predict_cell_type

        # Expression data where CD3D, CD3E are highly expressed (T cell markers)
        X = np.zeros((1, 10))
        X[0, 0] = 5.0  # CD3D
        X[0, 1] = 4.0  # CD3E

        adata = ad.AnnData(X=X)
        adata.var.index = [
            "CD3D", "CD3E", "CD19", "MS4A1", "CD14",
            "LYZ", "NKG7", "CD68", "COL1A1", "EPCAM"
        ]

        cell_type, confidence, alternatives = _predict_cell_type(
            embedding=np.zeros(512),
            adata=adata,
            cell_idx=0,
            valid_genes=list(adata.var.index),
        )

        result = {
            "predicted_type": cell_type,
            "confidence": confidence,
            "alternatives": alternatives,
        }
        assert result == snapshot

    def test_predict_cell_type_b_cell_snapshot(self, snapshot: SnapshotAssertion):
        """Snapshot test for B cell prediction."""
        from src.agent.scgpt.tools.annotate import _predict_cell_type

        # Expression data where CD19, MS4A1 are highly expressed (B cell markers)
        X = np.zeros((1, 10))
        X[0, 2] = 6.0  # CD19
        X[0, 3] = 5.0  # MS4A1

        adata = ad.AnnData(X=X)
        adata.var.index = [
            "CD3D", "CD3E", "CD19", "MS4A1", "CD14",
            "LYZ", "NKG7", "CD68", "COL1A1", "EPCAM"
        ]

        cell_type, confidence, alternatives = _predict_cell_type(
            embedding=np.zeros(512),
            adata=adata,
            cell_idx=0,
            valid_genes=list(adata.var.index),
        )

        result = {
            "predicted_type": cell_type,
            "confidence": confidence,
            "alternatives": alternatives,
        }
        assert result == snapshot

    def test_predict_cell_type_monocyte_snapshot(self, snapshot: SnapshotAssertion):
        """Snapshot test for Monocyte prediction."""
        from src.agent.scgpt.tools.annotate import _predict_cell_type

        # Expression data where CD14, LYZ are highly expressed (Monocyte markers)
        X = np.zeros((1, 10))
        X[0, 4] = 7.0  # CD14
        X[0, 5] = 6.0  # LYZ

        adata = ad.AnnData(X=X)
        adata.var.index = [
            "CD3D", "CD3E", "CD19", "MS4A1", "CD14",
            "LYZ", "NKG7", "CD68", "COL1A1", "EPCAM"
        ]

        cell_type, confidence, alternatives = _predict_cell_type(
            embedding=np.zeros(512),
            adata=adata,
            cell_idx=0,
            valid_genes=list(adata.var.index),
        )

        result = {
            "predicted_type": cell_type,
            "confidence": confidence,
            "alternatives": alternatives,
        }
        assert result == snapshot

    def test_batch_mixing_score_snapshot(self, snapshot: SnapshotAssertion):
        """Snapshot test for batch mixing score with known random state."""
        from src.agent.scgpt.tools.integrate import _compute_batch_mixing_score

        # Use fixed seed for reproducible results
        np.random.seed(123)
        embeddings = np.random.randn(50, 10)
        batch_labels = np.array([0, 1] * 25)

        score = _compute_batch_mixing_score(embeddings, batch_labels, n_neighbors=5)

        result = {"batch_mixing_score": round(score, 6)}
        assert result == snapshot

    @pytest.mark.asyncio
    async def test_embedding_result_structure_snapshot(
        self, deterministic_mock_loader, snapshot: SnapshotAssertion
    ):
        """Snapshot test for embedding result structure (not values)."""
        with patch("src.agent.scgpt.tools.embeddings.get_loader", return_value=deterministic_mock_loader):
            from src.agent.scgpt.tools.embeddings import get_gene_embeddings

            result = await get_gene_embeddings(
                gene_list=["TP53", "BRCA1", "MYC"],
                model_checkpoint="scGPT_human",
                include_similarity=True,
            )

            # Snapshot the structure, not the actual embedding values
            structure = {
                "num_embeddings": len(result.embeddings),
                "genes_found": [e.gene for e in result.embeddings],
                "embedding_dim": result.embedding_dim,
                "model_used": result.model_used,
                "has_similarity_matrix": result.similarity_matrix is not None,
                "similarity_matrix_shape": (
                    [len(result.similarity_matrix), len(result.similarity_matrix[0])]
                    if result.similarity_matrix
                    else None
                ),
            }
            assert structure == snapshot
