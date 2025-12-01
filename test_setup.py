"""Quick test to verify setup."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

# Test imports
print("Testing imports...")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
    else:
        print("  Using CPU")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    from src.config import config
    print(f"✓ Config loaded - Latent dim: {config['latent_CAE_2D']}")
except Exception as e:
    print(f"✗ Config import failed: {e}")

try:
    from src.nonlinear_reduction.load_data import load_projected_data
    print("✓ Data loading module imported")
except Exception as e:
    print(f"✗ Data loading import failed: {e}")

try:
    from src.nonlinear_reduction.CAE_model import build_cae_2d, get_device
    print("✓ CAE model module imported")
except Exception as e:
    print(f"✗ CAE model import failed: {e}")

print("\nSetup check complete!")

