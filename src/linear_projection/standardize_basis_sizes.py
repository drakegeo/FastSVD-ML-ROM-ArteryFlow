"""
Standardize POD basis sizes for all components to enable 3-kernel CAE training.

For a convolutional autoencoder with 3 input kernels (x, y, z), all basis matrices
must have the same number of modes. This script:
1. Checks current basis sizes
2. Truncates or pads all bases to a fixed target size (default: 256)
3. Saves standardized bases

Usage:
    python src/linear_projection/standardize_basis_sizes.py [--size SIZE]
"""
import sys
from pathlib import Path
import numpy as np
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try to import config for default target size
try:
    from src.config import config
    DEFAULT_TARGET_SIZE = config.get('target_basis_modes', 256)
except (ImportError, AttributeError, KeyError):
    DEFAULT_TARGET_SIZE = 256

def standardize_basis_sizes(target_size=None):
    """
    Standardize POD basis sizes for all components to a fixed target size.
    
    Parameters:
    -----------
    target_size : int, optional
        Target number of modes for all bases (default: from config or 256)
        - If a basis has more modes: truncate to target_size
        - If a basis has fewer modes: pad with zeros to target_size
    """
    if target_size is None:
        target_size = DEFAULT_TARGET_SIZE
    pod_basis_dir = Path('./data/POD_basis')
    svd_results_dir = Path('./data/SVD_results')
    
    if not pod_basis_dir.exists():
        print("Error: POD basis directory not found.")
        print("Please run POD basis calculation first:")
        print("  python src/linear_projection/process_all_components.py")
        return
    
    components = ['x', 'y', 'z']
    labels = ['X-component', 'Y-component', 'Z-component']
    
    print("=" * 80)
    print("STANDARDIZING POD BASIS SIZES")
    print("=" * 80)
    print()
    
    # Load all bases and check sizes
    basis_info = {}
    for comp, label in zip(components, labels):
        # Try both naming conventions
        pod_basis_file = pod_basis_dir / f'u{comp}.npy'
        if not pod_basis_file.exists():
            pod_basis_file = pod_basis_dir / f'Basis_u{comp}.npy'
        
        if not pod_basis_file.exists():
            print(f"  ✗ POD basis not found: {pod_basis_file}")
            continue
        
        pod_basis = np.load(pod_basis_file)
        num_nodes, num_modes = pod_basis.shape
        
        basis_info[comp] = {
            'label': label,
            'basis': pod_basis,
            'num_nodes': num_nodes,
            'num_modes': num_modes,
            'file': pod_basis_file
        }
        
        print(f"{label} ({comp.upper()}):")
        print(f"  Current size: ({num_nodes:,} nodes, {num_modes:,} modes)")
    
    if len(basis_info) != 3:
        print("\nError: Not all POD bases found. Cannot standardize.")
        return
    
    # Show current sizes
    sizes = [info['num_modes'] for info in basis_info.values()]
    min_size = min(sizes)
    max_size = max(sizes)
    
    print("\n" + "-" * 80)
    print("Current Basis Sizes:")
    for comp, info in basis_info.items():
        print(f"  {info['label']}: {info['num_modes']:,} modes")
    print(f"\n  Minimum: {min_size:,} modes")
    print(f"  Maximum: {max_size:,} modes")
    print(f"  Target size: {target_size:,} modes")
    print("-" * 80)
    
    common_size = target_size
    print(f"\nStandardizing all bases to: {common_size:,} modes")
    
    # Truncate all bases to common size
    print("\n" + "=" * 80)
    print("TRUNCATING BASES TO COMMON SIZE")
    print("=" * 80)
    print()
    
    # Backup original bases
    backup_dir = pod_basis_dir / 'backup_original'
    backup_dir.mkdir(exist_ok=True)
    print("Creating backups of original bases...")
    for comp, info in basis_info.items():
        # Use original filename pattern for backup
        original_name = info['file'].name
        backup_file = backup_dir / f'{original_name.replace(".npy", "_original.npy")}'
        np.save(backup_file, info['basis'])
        print(f"  Backed up {info['label']} to {backup_file.name}")
    
    # Truncate and save
    standardized_bases = {}
    for comp, info in basis_info.items():
        original_modes = info['num_modes']
        pod_basis = info['basis']
        
        if original_modes > common_size:
            # Truncate: keep first 'common_size' modes
            truncated_basis = pod_basis[:, :common_size]
            print(f"{info['label']}: Truncated from {original_modes:,} to {common_size:,} modes "
                  f"({(1 - common_size/original_modes)*100:.2f}% reduction)")
        elif original_modes < common_size:
            # Pad with zeros to reach target size
            padded_basis = np.zeros((info['num_nodes'], common_size), dtype=pod_basis.dtype)
            padded_basis[:, :original_modes] = pod_basis
            truncated_basis = padded_basis
            print(f"{info['label']}: Padded from {original_modes:,} to {common_size:,} modes "
                  f"(last {common_size - original_modes:,} modes are zero)")
        else:
            # Already correct size
            truncated_basis = pod_basis
            print(f"{info['label']}: Already {common_size:,} modes (no change)")
        
        # Save standardized basis
        np.save(info['file'], truncated_basis)
        standardized_bases[comp] = truncated_basis
        
        print(f"  ✓ Saved: {info['file'].name}, shape: {truncated_basis.shape}")
    
    print("\n" + "=" * 80)
    print("STANDARDIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nAll bases now have {common_size:,} modes")
    print("\nBasis shapes:")
    for comp, info in basis_info.items():
        # Use the standardized basis we already have in memory
        if comp in standardized_bases:
            print(f"  {info['label']}: {standardized_bases[comp].shape}")
        else:
            # Fallback: show expected shape
            print(f"  {info['label']}: ({info['num_nodes']:,} nodes, {common_size:,} modes)")
    
    print(f"\n✓ Bases are ready for 3-kernel CAE training!")
    print(f"\nNote: Original bases backed up to: {backup_dir}")
    print("\nNext step: Re-run projection script to use standardized bases:")
    print("  python src/linear_projection/project_solutions.py")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Standardize POD basis sizes for all components to a fixed size (default: 256)'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=None,
        help=f'Target number of modes for all bases (default: {DEFAULT_TARGET_SIZE} from config)'
    )
    
    args = parser.parse_args()
    standardize_basis_sizes(target_size=args.size)

