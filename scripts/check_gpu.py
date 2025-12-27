"""Check GPU availability and PyTorch compatibility."""

import subprocess
import sys

from loguru import logger


def main():
    logger.info("=" * 60)
    logger.info("GPU & PyTorch Diagnostics")
    logger.info("=" * 60)

    # Check Python version
    logger.info(f"Python version: {sys.version}")

    # Check PyTorch
    try:
        import torch
        import torch.version  # Explicit import for type checker

        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"PyTorch built with CUDA: {torch.version.cuda}")

        # CUDA availability
        logger.info(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
            logger.info(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

            # GPU count and details
            gpu_count = torch.cuda.device_count()
            logger.info(f"Number of GPUs: {gpu_count}")

            for i in range(gpu_count):
                logger.info(f"--- GPU {i} ---")
                logger.info(f"  Name: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  Compute capability: {props.major}.{props.minor}")
                logger.info(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
                logger.info(f"  Multi-processor count: {props.multi_processor_count}")

                # Current memory usage
                if torch.cuda.is_available():
                    torch.cuda.set_device(i)
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    logger.info(f"  Memory allocated: {allocated:.2f} GB")
                    logger.info(f"  Memory reserved: {reserved:.2f} GB")

            # Current device
            logger.info(f"Current device: {torch.cuda.current_device()}")
            logger.info(
                f"Current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}"
            )

            # Quick tensor test
            logger.info("--- Quick GPU Test ---")
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.randn(1000, 1000, device="cuda")
            z = torch.matmul(x, y)
            logger.success("Matrix multiplication test: SUCCESS")
            logger.info(f"Result tensor device: {z.device}")

        else:
            logger.warning("No CUDA-capable GPU detected or CUDA not properly installed.")
            logger.warning("PyTorch will use CPU only.")

            # Check if MPS (Apple Silicon) is available
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("MPS (Apple Silicon) is available!")
                logger.info(f"MPS built: {torch.backends.mps.is_built()}")

    except ImportError:
        logger.error("PyTorch is NOT installed.")
        logger.error("Install with: pip install torch torchvision torchaudio")
        logger.error("For CUDA support, visit: https://pytorch.org/get-started/locally/")

    # Also check for nvidia-smi
    logger.info("=" * 60)
    logger.info("NVIDIA System Management Interface (nvidia-smi)")
    logger.info("=" * 60)

    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            logger.info(f"\n{result.stdout}")
        else:
            logger.error(f"nvidia-smi failed: {result.stderr}")
    except FileNotFoundError:
        logger.warning("nvidia-smi not found. NVIDIA drivers may not be installed.")
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out.")
    except Exception as e:
        logger.error(f"Error running nvidia-smi: {e}")


if __name__ == "__main__":
    main()
