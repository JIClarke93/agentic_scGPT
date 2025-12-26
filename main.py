"""Entry point for the agentic-scgpt project."""

import argparse
import asyncio


def main():
    parser = argparse.ArgumentParser(
        description="Agentic scGPT - Single-cell analysis via MCP"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Server command
    server_parser = subparsers.add_parser("server", help="Run the MCP server")
    server_parser.add_argument(
        "--port", type=int, default=8000, help="Port to run server on"
    )

    # Agent command
    agent_parser = subparsers.add_parser("agent", help="Run the interactive agent")
    agent_parser.add_argument(
        "--server-url",
        default="http://localhost:8000/mcp",
        help="MCP server URL",
    )

    # GPU check command
    subparsers.add_parser("check-gpu", help="Check GPU availability")

    args = parser.parse_args()

    if args.command == "server":
        from src.server import mcp

        print(f"Starting scGPT MCP server on port {args.port}...")
        mcp.run(transport="streamable-http", port=args.port)

    elif args.command == "agent":
        from src.agent import run_interactive

        asyncio.run(run_interactive())

    elif args.command == "check-gpu":
        from scripts.check_gpu import main as check_gpu

        check_gpu()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
