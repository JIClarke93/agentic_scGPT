"""Agentic scGPT - Single-cell foundation model exposed via MCP."""

__version__ = "0.1.0"

# Main exports for convenient access
from .agent.scgpt import create_agent, ScGPTLoader, get_loader, SYSTEM_PROMPT
from .infra.mcp import mcp

__all__ = [
    "__version__",
    "create_agent",
    "ScGPTLoader",
    "get_loader",
    "SYSTEM_PROMPT",
    "mcp",
]
