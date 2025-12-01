"""
Test reconstruction error for projected test data.

This script:
1. Loads original test data from data/preprocessed/test_u{x,y,z}/
2. Loads projected test data from data/linear_projected/test_u{x,y,z}/
3. Loads POD basis matrices from data/POD_basis/u{x,y,z}.npy
4. Reconstructs original data from projected data: reconstructed = basis @ projected
5. Computes L2 reconstruction error: ||original - reconstructed||_F / ||original||_F

Usage:
    python src/linear_projection/test_reconstruction_error.py
"""
import sys
from pathlib import Path
import numpy as np
import natsort

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def compute_reconstruction_error():
    """
    Compute reconstruction error for test data after projection.
    """
    # Directories
    pod_basis_dir = Path('./data/POD_basis')
    preprocessed_dir = Path('./data/preprocessed')
    linear_projected_dir = Path('./data/linear_projected')
    
    if not pod_basis_dir.exists():
        print("Error: POD basis directory not found.")
        return
    
    if not linear_projected_dir.exists():
        print("Error: Linear projected directory not found.")
        print("Please run projection first:")
        print("  python src/linear_projection/project_solutions.py")
        return
    
    components = ['x', 'y', 'z']
    labels = ['X-component', 'Y-component', 'Z-component']
    
    print("=" * 80)
    print("TESTING RECONSTRUCTION ERROR")
    print("=" * 80)
    print()
    
    all_errors = {}
    
    # Process each component
    for comp, label in zip(components, labels):
        print(f"\n{'='*80}")
        print(f"Testing component: {comp.upper()}")
        print(f"{'='*80}\n")
        
        # Load POD basis
        pod_basis_file = pod_basis_dir / f'u{comp}.npy'
        if not pod_basis_file.exists():
            pod_basis_file = pod_basis_dir / f'Basis_u{comp}.npy'
        
        if not pod_basis_file.exists():
            print(f"  ✗ POD basis not found for component {comp}")
            continue
        
        pod_basis = np.load(pod_basis_file)  # Shape: (nodes, modes)
        num_nodes, num_modes = pod_basis.shape
        print(f"  POD basis shape: ({num_nodes:,} nodes, {num_modes:,} modes)")
        
        # Load original test data
        test_dir = preprocessed_dir / f'test_u{comp}'
        if not test_dir.exists():
            print(f"  ✗ Test data directory not found: {test_dir}")
            continue
        
        test_files = natsort.natsorted([f for f in test_dir.glob('*.npy')])
        if not test_files:
            print(f"  ✗ No test files found in {test_dir}")
            continue
        
        # Load projected test data from new folder structure
        projected_test_dir = linear_projected_dir / f'test_u{comp}'
        if not projected_test_dir.exists():
            print(f"  ✗ Projected test data directory not found: {projected_test_dir}")
            continue
        
        projected_test_files = natsort.natsorted([f for f in projected_test_dir.glob('*.npy')])
        if not projected_test_files:
            print(f"  ✗ No projected test files found in {projected_test_dir}")
            continue
        
        print(f"  Found {len(test_files)} original test simulation files")
        print(f"  Found {len(projected_test_files)} projected test simulation files")
        print()
        
        if len(projected_test_files) != len(test_files):
            print(f"  Warning: Mismatch - {len(projected_test_files)} projected test vs {len(test_files)} original test files")
            # Continue with minimum length
            num_files = min(len(projected_test_files), len(test_files))
            test_files = test_files[:num_files]
            projected_test_files = projected_test_files[:num_files]
        
        # Compute reconstruction errors
        errors = []
        for i, (test_file, projected_test_file) in enumerate(zip(test_files, projected_test_files)):
            # Load original test data
            original = np.load(test_file)  # Shape: (nodes, timesteps)
            
            # Load projected test data
            projected_test_data = np.load(projected_test_file)  # Shape: (modes, timesteps)
            # Load original test data
            original = np.load(test_file)  # Shape: (nodes, timesteps)
            
            # Reconstruct: basis @ projected
            # (nodes, modes) @ (modes, timesteps) = (nodes, timesteps)
            reconstructed = pod_basis @ projected_test_data  # Shape: (nodes, timesteps)
            
            # Check shapes match
            if original.shape != reconstructed.shape:
                print(f"    Warning: Shape mismatch for {test_file.name}")
                print(f"      Original: {original.shape}, Reconstructed: {reconstructed.shape}")
                continue
            
            # Compute L2 reconstruction error (Frobenius norm)
            error_norm = np.linalg.norm(original - reconstructed, ord='fro')
            original_norm = np.linalg.norm(original, ord='fro')
            relative_error = error_norm / original_norm if original_norm > 0 else 0.0
            
            errors.append(relative_error)
            
            print(f"  Test simulation {i+1} ({test_file.name}):")
            print(f"    Original shape: {original.shape}")
            print(f"    Reconstructed shape: {reconstructed.shape}")
            print(f"    Relative L2 error: {relative_error:.6e}")
        
        if errors:
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            min_error = np.min(errors)
            
            all_errors[comp] = {
                'mean': mean_error,
                'max': max_error,
                'min': min_error,
                'all': errors
            }
            
            print(f"\n  {label} Summary:")
            print(f"    Mean L2 error: {mean_error:.6e}")
            print(f"    Max L2 error:  {max_error:.6e}")
            print(f"    Min L2 error:  {min_error:.6e}")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    if all_errors:
        print("\nReconstruction Errors (L2 relative):")
        for comp, label in zip(components, labels):
            if comp in all_errors:
                err = all_errors[comp]
                print(f"  {label}:")
                print(f"    Mean: {err['mean']:.6e}")
                print(f"    Max:  {err['max']:.6e}")
                print(f"    Min:  {err['min']:.6e}")
        
        # Overall mean across all components
        overall_mean = np.mean([all_errors[c]['mean'] for c in components if c in all_errors])
        print(f"\n  Overall mean error (across all components): {overall_mean:.6e}")
    else:
        print("\n  No errors computed - check for missing files or errors above.")
    
    print()


if __name__ == '__main__':
    compute_reconstruction_error()

