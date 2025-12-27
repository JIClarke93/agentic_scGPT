"""Pydantic AI agent for single-cell genomics analysis.

This module provides the main agent interface for interacting with scGPT tools
via the Model Context Protocol (MCP). It supports both interactive CLI sessions
and programmatic single-query execution.

Example:
    >>> from src.agent.scgpt.agent import run_single_query
    >>> import asyncio
    >>> result = asyncio.run(run_single_query("What genes are similar to TP53?"))
"""

import asyncio
import sys

from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP

from .constants import DEFAULT_MCP_SERVER_URL, DEFAULT_MODEL
from .prompt import SYSTEM_PROMPT


def create_agent(mcp_server_url: str = DEFAULT_MCP_SERVER_URL) -> Agent:
    """Create a Pydantic AI agent connected to the scGPT MCP server.

    Initializes an agent with the scGPT system prompt and connects it to
    the MCP server providing single-cell analysis tools.

    Args:
        mcp_server_url: URL of the FastMCP server endpoint.

    Returns:
        Agent: Configured Pydantic AI agent with MCP toolset.
    """
    server = MCPServerStreamableHTTP(mcp_server_url)

    agent = Agent(
        DEFAULT_MODEL,
        toolsets=[server],
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


async def run_interactive(mcp_server_url: str = DEFAULT_MCP_SERVER_URL) -> None:
    """Run the agent in interactive command-line mode.

    Starts a REPL-style interface where users can submit queries and receive
    responses. The session continues until the user types 'quit' or presses Ctrl+C.

    Args:
        mcp_server_url: URL of the FastMCP server endpoint.
    """
    logger.info(f"Creating agent with MCP server at {mcp_server_url}")
    agent = create_agent(mcp_server_url)

    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write("scGPT Analysis Agent\n")
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write(f"Connected to MCP server at {mcp_server_url}\n")
    sys.stdout.write("Type 'quit' to exit\n\n")
    sys.stdout.flush()

    async with agent:
        logger.info("Agent session started")
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ("quit", "exit", "q"):
                    sys.stdout.write("Goodbye!\n")
                    logger.info("User requested exit")
                    break

                if not user_input:
                    continue

                logger.debug(f"Processing user query: {user_input[:50]}...")
                result = await agent.run(user_input)
                sys.stdout.write(f"\nAssistant: {result.output}\n")
                sys.stdout.flush()

            except KeyboardInterrupt:
                sys.stdout.write("\nGoodbye!\n")
                logger.info("Session interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                sys.stdout.write(f"\nError: {e}\n")
                sys.stdout.flush()


async def run_single_query(
    query: str,
    mcp_server_url: str = DEFAULT_MCP_SERVER_URL,
) -> str:
    """Run a single query against the agent.

    Executes a one-off query without maintaining a persistent session.
    Useful for programmatic access or batch processing.

    Args:
        query: The user's question or analysis request.
        mcp_server_url: URL of the FastMCP server endpoint.

    Returns:
        str: The agent's text response.
    """
    logger.info(f"Running single query against {mcp_server_url}")
    logger.debug(f"Query: {query[:100]}...")
    agent = create_agent(mcp_server_url)

    async with agent:
        result = await agent.run(query)
        logger.info("Query completed successfully")
        return result.output


async def example_session() -> None:
    """Demonstrate example agent interactions.

    Runs a series of predefined queries to showcase the agent's capabilities
    for cell type annotation, batch integration, and gene similarity analysis.
    """
    logger.info("Starting example session")
    agent = create_agent()

    examples = [
        "What cell types are in my PBMC dataset at data/pbmc_3k.h5ad?",
        "I have three batches of data. Can you integrate them?",
        "What genes are most similar to TP53 based on scGPT embeddings?",
    ]

    async with agent:
        for query in examples:
            logger.info(f"Running example query: {query[:50]}...")
            sys.stdout.write(f"\n{'='*60}\n")
            sys.stdout.write(f"Query: {query}\n")
            sys.stdout.write("=" * 60 + "\n")
            result = await agent.run(query)
            sys.stdout.write(f"Response: {result.output}\n")
            sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(run_interactive())
