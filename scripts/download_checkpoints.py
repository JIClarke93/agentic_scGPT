"""Download scGPT pretrained checkpoints from Hugging Face or direct URLs."""

import argparse
import hashlib
import sys
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn

console = Console()

# scGPT checkpoint configurations
# These are available from the scGPT repository/releases
CHECKPOINTS = {
    "scGPT_human": {
        "url": "https://huggingface.co/bowang/scGPT/resolve/main/scGPT_human/best_model.pt",
        "filename": "best_model.pt",
        "subdir": "scGPT_human",
        "description": "scGPT pretrained on human cells (33M cells from CellXGene)",
        "vocab_url": "https://huggingface.co/bowang/scGPT/resolve/main/scGPT_human/vocab.json",
    },
    "scGPT_CP": {
        "url": "https://huggingface.co/bowang/scGPT/resolve/main/scGPT_CP/best_model.pt",
        "filename": "best_model.pt",
        "subdir": "scGPT_CP",
        "description": "scGPT for cell perturbation prediction",
        "vocab_url": "https://huggingface.co/bowang/scGPT/resolve/main/scGPT_CP/vocab.json",
    },
    "scGPT_BC": {
        "url": "https://huggingface.co/bowang/scGPT/resolve/main/scGPT_BC/best_model.pt",
        "filename": "best_model.pt",
        "subdir": "scGPT_BC",
        "description": "scGPT for batch correction",
        "vocab_url": "https://huggingface.co/bowang/scGPT/resolve/main/scGPT_BC/vocab.json",
    },
}


def get_checkpoint_dir() -> Path:
    """Get the checkpoints directory, creating it if necessary."""
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir


def download_file(url: str, dest: Path, description: str = "Downloading") -> bool:
    """Download a file with progress bar."""
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(description, total=total)

                with open(dest, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))
        return True
    except httpx.HTTPStatusError as e:
        console.print(f"[red]HTTP error: {e.response.status_code}[/red]")
        return False
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        return False


def download_checkpoint(name: str, force: bool = False) -> bool:
    """Download a specific checkpoint."""
    if name not in CHECKPOINTS:
        console.print(f"[red]Unknown checkpoint: {name}[/red]")
        console.print(f"Available checkpoints: {', '.join(CHECKPOINTS.keys())}")
        return False

    config = CHECKPOINTS[name]
    checkpoint_dir = get_checkpoint_dir() / config["subdir"]
    checkpoint_dir.mkdir(exist_ok=True)

    model_path = checkpoint_dir / config["filename"]
    vocab_path = checkpoint_dir / "vocab.json"

    console.print(f"\n[bold blue]{name}[/bold blue]: {config['description']}")

    # Download model weights
    if model_path.exists() and not force:
        console.print(f"  [yellow]Model already exists:[/yellow] {model_path}")
    else:
        console.print(f"  [cyan]Downloading model weights...[/cyan]")
        if not download_file(config["url"], model_path, f"  {config['filename']}"):
            return False
        console.print(f"  [green]Saved to:[/green] {model_path}")

    # Download vocabulary
    if "vocab_url" in config:
        if vocab_path.exists() and not force:
            console.print(f"  [yellow]Vocab already exists:[/yellow] {vocab_path}")
        else:
            console.print(f"  [cyan]Downloading vocabulary...[/cyan]")
            if not download_file(config["vocab_url"], vocab_path, "  vocab.json"):
                console.print("  [yellow]Warning: Could not download vocab file[/yellow]")
            else:
                console.print(f"  [green]Saved to:[/green] {vocab_path}")

    return True


def list_checkpoints():
    """List available checkpoints and their status."""
    checkpoint_dir = get_checkpoint_dir()

    console.print("\n[bold]Available scGPT Checkpoints:[/bold]\n")

    for name, config in CHECKPOINTS.items():
        model_path = checkpoint_dir / config["subdir"] / config["filename"]
        status = "[green]Downloaded[/green]" if model_path.exists() else "[dim]Not downloaded[/dim]"
        console.print(f"  [bold]{name}[/bold] - {status}")
        console.print(f"    {config['description']}")
        console.print()


def main():
    parser = argparse.ArgumentParser(
        description="Download scGPT pretrained checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.download_checkpoints --list
  python -m scripts.download_checkpoints scGPT_human
  python -m scripts.download_checkpoints --all
  python -m scripts.download_checkpoints scGPT_human --force
        """,
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        choices=list(CHECKPOINTS.keys()),
        help="Checkpoint to download",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available checkpoints",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available checkpoints",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )

    args = parser.parse_args()

    console.print("[bold]scGPT Checkpoint Downloader[/bold]")
    console.print(f"Checkpoint directory: {get_checkpoint_dir()}\n")

    if args.list:
        list_checkpoints()
        return 0

    if args.all:
        console.print("[bold]Downloading all checkpoints...[/bold]")
        success = True
        for name in CHECKPOINTS:
            if not download_checkpoint(name, args.force):
                success = False
        return 0 if success else 1

    if args.checkpoint:
        success = download_checkpoint(args.checkpoint, args.force)
        return 0 if success else 1

    # Default: show help and list
    parser.print_help()
    list_checkpoints()
    return 0


if __name__ == "__main__":
    sys.exit(main())
