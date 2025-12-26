"""Check GPU availability and PyTorch compatibility."""

import subprocess
import sys


def main():
    print("=" * 60)
    print("GPU & PyTorch Diagnostics")
    print("=" * 60)

    # Check Python version
    print(f"\nPython version: {sys.version}")

    # Check PyTorch
    try:
        import torch

        print(f"\nPyTorch version: {torch.__version__}")
        print(f"PyTorch built with CUDA: {torch.version.cuda}")

        # CUDA availability
        print(f"\nCUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

            # GPU count and details
            gpu_count = torch.cuda.device_count()
            print(f"\nNumber of GPUs: {gpu_count}")

            for i in range(gpu_count):
                print(f"\n--- GPU {i} ---")
                print(f"  Name: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"  Compute capability: {props.major}.{props.minor}")
                print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"  Multi-processor count: {props.multi_processor_count}")

                # Current memory usage
                if torch.cuda.is_available():
                    torch.cuda.set_device(i)
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"  Memory allocated: {allocated:.2f} GB")
                    print(f"  Memory reserved: {reserved:.2f} GB")

            # Current device
            print(f"\nCurrent device: {torch.cuda.current_device()}")
            print(
                f"Current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}"
            )

            # Quick tensor test
            print("\n--- Quick GPU Test ---")
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.randn(1000, 1000, device="cuda")
            z = torch.matmul(x, y)
            print("Matrix multiplication test: SUCCESS")
            print(f"Result tensor device: {z.device}")

        else:
            print("\nNo CUDA-capable GPU detected or CUDA not properly installed.")
            print("PyTorch will use CPU only.")

            # Check if MPS (Apple Silicon) is available
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                print("\nMPS (Apple Silicon) is available!")
                print(f"MPS built: {torch.backends.mps.is_built()}")

    except ImportError:
        print("\nPyTorch is NOT installed.")
        print("Install with: pip install torch torchvision torchaudio")
        print("For CUDA support, visit: https://pytorch.org/get-started/locally/")

    # Also check for nvidia-smi
    print("\n" + "=" * 60)
    print("NVIDIA System Management Interface (nvidia-smi)")
    print("=" * 60)

    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("nvidia-smi failed:", result.stderr)
    except FileNotFoundError:
        print("nvidia-smi not found. NVIDIA drivers may not be installed.")
    except subprocess.TimeoutExpired:
        print("nvidia-smi timed out.")
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")


if __name__ == "__main__":
    main()
