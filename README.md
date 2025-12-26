# Agentic scGPT

A Pydantic AI agent that exposes [scGPT](https://github.com/bowang-lab/scGPT) single-cell foundation model capabilities as tools via the Model Context Protocol (MCP).

## Project Overview

This project bridges the gap between conversational AI agents and single-cell genomics analysis by:

1. **Exposing scGPT functions as MCP tools** via [FastMCP](https://github.com/jlowin/fastmcp)
2. **Creating a Pydantic AI agent** that can orchestrate these tools
3. **Adding reliability with Temporal** for retry logic and workflow orchestration

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Pydantic AI Agent                          │
│                    (Orchestration Layer)                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │ MCP Protocol
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastMCP Server                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Cell Type   │  │ Batch       │  │ Gene Network            │ │
│  │ Annotation  │  │ Integration │  │ Inference               │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Temporal Workflows                            │
│              (Retry Logic & Orchestration)                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      scGPT Model                                │
│              (GPU-accelerated inference)                        │
└─────────────────────────────────────────────────────────────────┘
```

## Planned MCP Tools (Phase 1)

### Tool 1: `annotate_cell_types`
Annotate single cells based on gene expression profiles using scGPT embeddings.

```python
@mcp.tool
async def annotate_cell_types(
    expression_data: str,  # Path to h5ad file or CSV
    reference_dataset: str = "cellxgene",
    batch_size: int = 64
) -> dict:
    """
    Annotate cell types from single-cell RNA-seq data.

    Returns predicted cell types with confidence scores.
    """
```

### Tool 2: `integrate_batches`
Integrate multiple scRNA-seq datasets while correcting for batch effects.

```python
@mcp.tool
async def integrate_batches(
    dataset_paths: list[str],  # Paths to multiple h5ad files
    batch_key: str = "batch",
    n_hvg: int = 2000
) -> dict:
    """
    Integrate multiple single-cell datasets with batch correction.

    Returns path to integrated dataset and integration metrics.
    """
```

### Tool 3: `get_gene_embeddings`
Extract gene embeddings for downstream analysis or gene network inference.

```python
@mcp.tool
async def get_gene_embeddings(
    gene_list: list[str],
    model_checkpoint: str = "scGPT_human"
) -> dict:
    """
    Get scGPT embeddings for a list of genes.

    Returns gene embeddings that can be used for similarity analysis.
    """
```

## Development Phases

### Phase 1: Foundation Setup
- [x] Set up project structure with `uv`
- [x] Install core dependencies (scGPT, FastMCP, Pydantic AI)
- [x] Verify GPU/CUDA compatibility
- [x] Download scGPT pretrained checkpoints

### Phase 2: MCP Server Implementation
- [x] Create FastMCP server skeleton
- [x] Implement `annotate_cell_types` tool
- [x] Implement `integrate_batches` tool
- [x] Implement `get_gene_embeddings` tool
- [x] Add input validation with Pydantic models
- [x] Write tool tests with [syrupy](https://github.com/syrupy-project/syrupy) snapshot testing

### Phase 3: Temporal Integration
- [ ] Set up Temporal worker
- [ ] Wrap scGPT operations as Temporal activities
- [ ] Add retry policies for GPU operations
- [ ] Implement workflow for multi-step analyses

### Phase 4: Pydantic AI Agent
- [ ] Create agent with MCP server connection
- [ ] Define system prompts for biology domain
- [ ] Add conversation memory/context
- [ ] Implement example analysis workflows

### Phase 5: Testing & Documentation
- [ ] Integration tests with sample data
- [ ] Performance benchmarks
- [ ] API documentation
- [ ] Usage examples and tutorials

## Installation

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (recommended)
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd agentic_scgpt

# Create virtual environment and install dependencies
uv sync

# Download scGPT checkpoints
python -m scripts.download_checkpoints

# Start Temporal (if using retry logic)
temporal server start-dev

# Run the MCP server
uv run python -m src.server
```

## Project Structure

```
agentic_scgpt/
├── src/
│   ├── __init__.py
│   ├── server.py           # FastMCP server with scGPT tools
│   ├── agent.py            # Pydantic AI agent
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── annotate.py     # Cell type annotation tool
│   │   ├── integrate.py    # Batch integration tool
│   │   └── embeddings.py   # Gene embeddings tool
│   ├── workflows/
│   │   ├── __init__.py
│   │   └── activities.py   # Temporal activities
│   └── models/
│       ├── __init__.py
│       └── schemas.py      # Pydantic schemas for tool I/O
├── scripts/
│   ├── check_gpu.py        # GPU diagnostics
│   └── download_checkpoints.py
├── tests/
├── checkpoints/            # scGPT model weights
├── pyproject.toml
└── README.md
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `scgpt` | Single-cell foundation model |
| `fastmcp` | MCP server framework |
| `pydantic-ai` | AI agent framework |
| `temporalio` | Workflow orchestration & retries |
| `torch` | Deep learning backend |
| `scanpy` | Single-cell analysis utilities |
| `anndata` | Data structures for scRNA-seq |

## Example Usage

### Running the MCP Server

```python
# src/server.py
from fastmcp import FastMCP

mcp = FastMCP("scGPT Tools")

@mcp.tool
async def annotate_cell_types(expression_data: str) -> dict:
    """Annotate cell types from scRNA-seq data."""
    # Implementation here
    pass

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8000)
```

### Connecting the Pydantic AI Agent

```python
# src/agent.py
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP

server = MCPServerStreamableHTTP("http://localhost:8000/mcp")

agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    toolsets=[server],
    system_prompt="""You are a single-cell genomics expert assistant.
    You help researchers analyze scRNA-seq data using scGPT tools.
    Always explain your analysis steps and interpret results."""
)

async def main():
    async with agent:
        result = await agent.run(
            "Annotate the cell types in my dataset at data/pbmc.h5ad"
        )
        print(result.output)
```

### With Temporal Retries

```python
# src/workflows/activities.py
from temporalio import activity
from datetime import timedelta

@activity.defn
async def annotate_cells_activity(data_path: str) -> dict:
    """Activity with automatic retry on GPU OOM errors."""
    # scGPT annotation logic
    pass

# In workflow
await workflow.execute_activity(
    annotate_cells_activity,
    "data/sample.h5ad",
    start_to_close_timeout=timedelta(minutes=10),
    retry_policy=RetryPolicy(
        maximum_attempts=3,
        initial_interval=timedelta(seconds=5),
        non_retryable_error_types=["InvalidDataError"]
    )
)
```

## GPU Requirements

scGPT requires a CUDA-capable GPU for efficient inference. Run the GPU check script:

```bash
uv run python scripts/check_gpu.py
```

Recommended specifications:
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8+
- cuDNN 8.6+

## Resources

- [scGPT GitHub](https://github.com/bowang-lab/scGPT)
- [scGPT Paper (Nature Methods)](https://www.nature.com/articles/s41592-024-02201-0)
- [FastMCP Documentation](https://gofastmcp.com/)
- [Pydantic AI MCP Docs](https://ai.pydantic.dev/mcp/overview/)
- [Temporal Python SDK](https://docs.temporal.io/develop/python)

## License

MIT
