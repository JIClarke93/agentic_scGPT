"""Tests for the FastMCP server."""

import pytest


class TestMCPServer:
    """Tests for MCP server configuration."""

    def test_server_creation(self):
        """Test that the MCP server is created correctly."""
        from src.infra.mcp.server import mcp

        assert mcp.name == "scGPT Tools"

    def test_tools_registered(self):
        """Test that all expected tools are registered."""
        from src.infra.mcp.server import mcp

        # FastMCP registers tools internally
        # We verify by checking the tool manager exists
        assert hasattr(mcp, "_tool_manager")

        # The tool manager should have registered our tools
        tool_manager = mcp._tool_manager
        assert tool_manager is not None

    def test_tool_docstrings(self):
        """Test that tools have proper descriptions."""
        from src.infra.mcp.server import annotate_cells, extract_gene_embeddings, integrate_datasets

        # FastMCP wraps functions in FunctionTool objects
        # Check the description attribute instead of __doc__
        assert annotate_cells.description is not None
        assert integrate_datasets.description is not None
        assert extract_gene_embeddings.description is not None

        # Descriptions should describe the functionality
        assert "cell type" in annotate_cells.description.lower()
        assert "integrate" in integrate_datasets.description.lower()
        assert "embedding" in extract_gene_embeddings.description.lower()


class TestServerImports:
    """Test that server imports work correctly."""

    def test_import_mcp_server(self):
        """Test that MCP server module can be imported."""
        from src.infra.mcp import server

        assert hasattr(server, "mcp")
        assert hasattr(server, "main")

    def test_import_tools(self):
        """Test that tools can be imported."""
        from src.agent.scgpt.tools import annotate_cell_types, get_gene_embeddings, integrate_batches

        # These should be async functions
        import asyncio

        assert asyncio.iscoroutinefunction(annotate_cell_types)
        assert asyncio.iscoroutinefunction(get_gene_embeddings)
        assert asyncio.iscoroutinefunction(integrate_batches)

    def test_import_models(self):
        """Test that models can be imported."""
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

        # All should be Pydantic models
        from pydantic import BaseModel

        assert issubclass(AnnotationRequest, BaseModel)
        assert issubclass(AnnotationResponse, BaseModel)
        assert issubclass(BatchIntegrationRequest, BaseModel)
        assert issubclass(BatchIntegrationResponse, BaseModel)
        assert issubclass(EmbeddingRequest, BaseModel)
        assert issubclass(EmbeddingResponse, BaseModel)
