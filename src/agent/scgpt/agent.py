"""Pydantic AI agent for single-cell genomics analysis."""

import asyncio

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP

from .prompt import SYSTEM_PROMPT


def create_agent(mcp_server_url: str = "http://localhost:8000/mcp") -> Agent:
    """
    Create a Pydantic AI agent connected to the scGPT MCP server.

    Args:
        mcp_server_url: URL of the FastMCP server

    Returns:
        Configured Pydantic AI agent
    """
    server = MCPServerStreamableHTTP(mcp_server_url)

    agent = Agent(
        "anthropic:claude-sonnet-4-20250514",
        toolsets=[server],
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


async def run_interactive(mcp_server_url: str = "http://localhost:8000/mcp"):
    """Run the agent in interactive mode."""
    agent = create_agent(mcp_server_url)

    print("=" * 60)
    print("scGPT Analysis Agent")
    print("=" * 60)
    print(f"Connected to MCP server at {mcp_server_url}")
    print("Type 'quit' to exit\n")

    async with agent:
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                result = await agent.run(user_input)
                print(f"\nAssistant: {result.output}")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")


async def run_single_query(
    query: str,
    mcp_server_url: str = "http://localhost:8000/mcp",
) -> str:
    """
    Run a single query against the agent.

    Args:
        query: The user's question or request
        mcp_server_url: URL of the FastMCP server

    Returns:
        The agent's response
    """
    agent = create_agent(mcp_server_url)

    async with agent:
        result = await agent.run(query)
        return result.output


async def example_session():
    """Demonstrate example agent interactions."""
    agent = create_agent()

    examples = [
        "What cell types are in my PBMC dataset at data/pbmc_3k.h5ad?",
        "I have three batches of data. Can you integrate them?",
        "What genes are most similar to TP53 based on scGPT embeddings?",
    ]

    async with agent:
        for query in examples:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print("=" * 60)
            result = await agent.run(query)
            print(f"Response: {result.output}")


if __name__ == "__main__":
    asyncio.run(run_interactive())
