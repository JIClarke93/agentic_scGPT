"""Tests for Pydantic models with snapshot testing."""

import pytest
from pydantic import ValidationError
from syrupy.assertion import SnapshotAssertion

from src.agent.scgpt.models import (
    AnnotationRequest,
    AnnotationResponse,
    BatchIntegrationRequest,
    BatchIntegrationResponse,
    CellAnnotation,
    EmbeddingRequest,
    EmbeddingResponse,
    GeneEmbedding,
)


class TestAnnotationSchemas:
    """Tests for annotation-related schemas."""

    def test_annotation_request_valid(self):
        req = AnnotationRequest(expression_data="path/to/data.h5ad")
        assert req.expression_data == "path/to/data.h5ad"
        assert req.reference_dataset == "cellxgene"
        assert req.batch_size == 64

    def test_annotation_request_custom_values(self):
        req = AnnotationRequest(
            expression_data="data.h5ad",
            reference_dataset="pbmc",
            batch_size=128,
        )
        assert req.reference_dataset == "pbmc"
        assert req.batch_size == 128

    def test_annotation_request_batch_size_bounds(self):
        # Valid lower bound
        req = AnnotationRequest(expression_data="data.h5ad", batch_size=1)
        assert req.batch_size == 1

        # Valid upper bound
        req = AnnotationRequest(expression_data="data.h5ad", batch_size=512)
        assert req.batch_size == 512

        # Invalid - too low
        with pytest.raises(ValidationError):
            AnnotationRequest(expression_data="data.h5ad", batch_size=0)

        # Invalid - too high
        with pytest.raises(ValidationError):
            AnnotationRequest(expression_data="data.h5ad", batch_size=1000)

    def test_cell_annotation_valid(self):
        ann = CellAnnotation(
            cell_id="cell_001",
            predicted_type="T cell",
            confidence=0.95,
        )
        assert ann.cell_id == "cell_001"
        assert ann.predicted_type == "T cell"
        assert ann.confidence == 0.95
        assert ann.alternative_types == []

    def test_cell_annotation_with_alternatives(self):
        ann = CellAnnotation(
            cell_id="cell_001",
            predicted_type="T cell",
            confidence=0.85,
            alternative_types=[("NK cell", 0.10), ("B cell", 0.05)],
        )
        assert len(ann.alternative_types) == 2

    def test_cell_annotation_confidence_bounds(self):
        # Valid bounds
        CellAnnotation(cell_id="c1", predicted_type="T", confidence=0.0)
        CellAnnotation(cell_id="c1", predicted_type="T", confidence=1.0)

        # Invalid
        with pytest.raises(ValidationError):
            CellAnnotation(cell_id="c1", predicted_type="T", confidence=-0.1)
        with pytest.raises(ValidationError):
            CellAnnotation(cell_id="c1", predicted_type="T", confidence=1.1)

    def test_annotation_response(self):
        result = AnnotationResponse(
            total_cells=100,
            annotations=[
                CellAnnotation(cell_id="c1", predicted_type="T cell", confidence=0.9)
            ],
            unique_types=["T cell"],
            output_path="/path/to/output.h5ad",
        )
        assert result.total_cells == 100
        assert len(result.annotations) == 1
        assert result.unique_types == ["T cell"]

    # Snapshot tests for serialization consistency
    def test_annotation_request_snapshot(self, snapshot: SnapshotAssertion):
        """Snapshot test for AnnotationRequest serialization."""
        req = AnnotationRequest(
            expression_data="data/sample.h5ad",
            reference_dataset="cellxgene",
            batch_size=64,
        )
        assert req.model_dump() == snapshot

    def test_cell_annotation_snapshot(self, snapshot: SnapshotAssertion):
        """Snapshot test for CellAnnotation with alternatives."""
        ann = CellAnnotation(
            cell_id="cell_001",
            predicted_type="T cell",
            confidence=0.92,
            alternative_types=[("NK cell", 0.05), ("B cell", 0.03)],
        )
        assert ann.model_dump() == snapshot

    def test_annotation_result_snapshot(self, snapshot: SnapshotAssertion):
        """Snapshot test for complete AnnotationResponse."""
        result = AnnotationResponse(
            total_cells=3,
            annotations=[
                CellAnnotation(
                    cell_id="cell_001",
                    predicted_type="T cell",
                    confidence=0.92,
                    alternative_types=[("NK cell", 0.05)],
                ),
                CellAnnotation(
                    cell_id="cell_002",
                    predicted_type="B cell",
                    confidence=0.87,
                    alternative_types=[("Plasma cell", 0.08)],
                ),
                CellAnnotation(
                    cell_id="cell_003",
                    predicted_type="Monocyte",
                    confidence=0.95,
                    alternative_types=[],
                ),
            ],
            unique_types=["B cell", "Monocyte", "T cell"],
            output_path="/data/annotated.h5ad",
        )
        assert result.model_dump() == snapshot


class TestBatchIntegrationSchemas:
    """Tests for batch integration schemas."""

    def test_integration_request_valid(self):
        req = BatchIntegrationRequest(
            dataset_paths=["data1.h5ad", "data2.h5ad"]
        )
        assert len(req.dataset_paths) == 2
        assert req.batch_key == "batch"
        assert req.n_hvg == 2000

    def test_integration_request_min_datasets(self):
        # Valid - 2 datasets
        BatchIntegrationRequest(dataset_paths=["a.h5ad", "b.h5ad"])

        # Invalid - only 1 dataset
        with pytest.raises(ValidationError):
            BatchIntegrationRequest(dataset_paths=["a.h5ad"])

    def test_integration_request_hvg_bounds(self):
        # Valid bounds
        BatchIntegrationRequest(
            dataset_paths=["a.h5ad", "b.h5ad"],
            n_hvg=500,
        )
        BatchIntegrationRequest(
            dataset_paths=["a.h5ad", "b.h5ad"],
            n_hvg=5000,
        )

        # Invalid
        with pytest.raises(ValidationError):
            BatchIntegrationRequest(
                dataset_paths=["a.h5ad", "b.h5ad"],
                n_hvg=100,
            )

    def test_integration_result(self):
        result = BatchIntegrationResponse(
            integrated_path="/path/to/integrated.h5ad",
            n_cells=10000,
            n_batches=3,
            batch_mixing_score=0.85,
            silhouette_score=0.72,
        )
        assert result.n_cells == 10000
        assert result.n_batches == 3

    # Snapshot tests
    def test_integration_request_snapshot(self, snapshot: SnapshotAssertion):
        """Snapshot test for BatchIntegrationRequest."""
        req = BatchIntegrationRequest(
            dataset_paths=["batch1.h5ad", "batch2.h5ad", "batch3.h5ad"],
            batch_key="sample_id",
            n_hvg=3000,
            output_path="/output/integrated.h5ad",
        )
        assert req.model_dump() == snapshot

    def test_integration_result_snapshot(self, snapshot: SnapshotAssertion):
        """Snapshot test for BatchIntegrationResponse."""
        result = BatchIntegrationResponse(
            integrated_path="/data/integrated.h5ad",
            n_cells=15000,
            n_batches=3,
            batch_mixing_score=0.85,
            silhouette_score=0.72,
        )
        assert result.model_dump() == snapshot


class TestEmbeddingSchemas:
    """Tests for embedding schemas."""

    def test_embedding_request_valid(self):
        req = EmbeddingRequest(gene_list=["TP53", "BRCA1"])
        assert len(req.gene_list) == 2
        assert req.model_checkpoint == "scGPT_human"
        assert req.include_similarity is False

    def test_embedding_request_empty_list(self):
        with pytest.raises(ValidationError):
            EmbeddingRequest(gene_list=[])

    def test_gene_embedding(self):
        emb = GeneEmbedding(
            gene="TP53",
            embedding=[0.1, 0.2, 0.3],
        )
        assert emb.gene == "TP53"
        assert len(emb.embedding) == 3

    def test_embedding_result(self):
        result = EmbeddingResponse(
            embeddings=[
                GeneEmbedding(gene="TP53", embedding=[0.1] * 512),
                GeneEmbedding(gene="BRCA1", embedding=[0.2] * 512),
            ],
            embedding_dim=512,
            model_used="scGPT_human",
        )
        assert len(result.embeddings) == 2
        assert result.embedding_dim == 512
        assert result.similarity_matrix is None

    def test_embedding_result_with_similarity(self):
        result = EmbeddingResponse(
            embeddings=[
                GeneEmbedding(gene="TP53", embedding=[0.1] * 512),
            ],
            embedding_dim=512,
            model_used="scGPT_human",
            similarity_matrix=[[1.0]],
        )
        assert result.similarity_matrix is not None

    # Snapshot tests
    def test_embedding_request_snapshot(self, snapshot: SnapshotAssertion):
        """Snapshot test for EmbeddingRequest."""
        req = EmbeddingRequest(
            gene_list=["TP53", "BRCA1", "MYC", "EGFR"],
            model_checkpoint="scGPT_human",
            include_similarity=True,
        )
        assert req.model_dump() == snapshot

    def test_gene_embedding_snapshot(self, snapshot: SnapshotAssertion):
        """Snapshot test for GeneEmbedding with small embedding."""
        emb = GeneEmbedding(
            gene="TP53",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],  # Small for readable snapshot
        )
        assert emb.model_dump() == snapshot

    def test_embedding_result_with_similarity_snapshot(self, snapshot: SnapshotAssertion):
        """Snapshot test for EmbeddingResponse with similarity matrix."""
        result = EmbeddingResponse(
            embeddings=[
                GeneEmbedding(gene="TP53", embedding=[0.1, 0.2, 0.3]),
                GeneEmbedding(gene="BRCA1", embedding=[0.4, 0.5, 0.6]),
                GeneEmbedding(gene="MYC", embedding=[0.7, 0.8, 0.9]),
            ],
            embedding_dim=3,
            model_used="scGPT_human",
            similarity_matrix=[
                [1.0, 0.85, 0.72],
                [0.85, 1.0, 0.68],
                [0.72, 0.68, 1.0],
            ],
        )
        assert result.model_dump() == snapshot
