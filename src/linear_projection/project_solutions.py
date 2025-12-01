"""
Project solutions onto POD basis matrices for all components.

For each simulation file (nodes × timesteps), projects it onto the POD basis
to get compressed representation (number_of_modes × timesteps).

Result: Each simulation is compressed from (nodes, timesteps) to (k, timesteps) 
where k is the number of POD modes in the basis.

Output structure matches preprocessed data:
    data/linear_projected/train_u{x,y,z}/sim_u*.npy
    data/linear_projected/test_u{x,y,z}/sim_u*.npy

Usage:
    python src/linear_projection/project_solutions.py
"""
import sys
from pathlib import Path
import numpy as np
import natsort

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def project_solutions():
    """
    Project all solution files onto their respective POD basis matrices.
    """
    # Directories
    pod_basis_dir = Path('./data/POD_basis')
    preprocessed_dir = Path('./data/preprocessed')
    linear_projected_dir = Path('./data/linear_projected')
    
    # Create output directory
    linear_projected_dir.mkdir(parents=True, exist_ok=True)
    
    if not pod_basis_dir.exists():
        print("Error: POD basis directory not found.")
        print("Please run POD basis calculation first:")
        print("  python src/linear_projection/process_all_components.py")
        return
    
    components = ['x', 'y', 'z']
    data_types = ['train', 'test']
    
    print("=" * 80)
    print("PROJECTING SOLUTIONS ONTO POD BASIS")
    print("=" * 80)
    print()
    
    # Process each component
    for comp in components:
        print(f"\n{'='*80}")
        print(f"Processing component: {comp.upper()}")
        print(f"{'='*80}\n")
        
        # Load POD basis (try both naming conventions)
        pod_basis_file = pod_basis_dir / f'u{comp}.npy'
        if not pod_basis_file.exists():
            pod_basis_file = pod_basis_dir / f'Basis_u{comp}.npy'
        if not pod_basis_file.exists():
            print(f"  ✗ POD basis not found for component {comp}")
            continue
        
        pod_basis = np.load(pod_basis_file)  # Shape: (nodes, k)
        num_nodes, num_modes = pod_basis.shape
        print(f"  POD basis shape: ({num_nodes:,} nodes, {num_modes:,} modes)")
        print(f"  Compression: {num_nodes:,} → {num_modes:,} ({num_nodes/num_modes:.2f}:1 ratio)")
        print()
        
        # Process train and test data separately
        for data_type in data_types:
            input_dir = preprocessed_dir / f'{data_type}_u{comp}'
            
            if not input_dir.exists():
                print(f"  Warning: Input directory not found: {input_dir}")
                continue
            
            # Create output directory matching preprocessed structure
            output_dir = linear_projected_dir / f'{data_type}_u{comp}'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all simulation files
            sim_files = natsort.natsorted([f for f in input_dir.glob('*.npy')])
            
            if not sim_files:
                print(f"  No simulation files found in {input_dir}")
                continue
            
            print(f"  Processing {data_type.upper()} data: {len(sim_files)} simulations")
            
            num_saved = 0
            for sim_file in sim_files:
                # Load solution: shape (nodes, timesteps)
                solution = np.load(sim_file)  # Shape: (nodes, timesteps)
                
                if solution.shape[0] != num_nodes:
                    print(f"    Warning: {sim_file.name} has {solution.shape[0]} nodes, "
                          f"expected {num_nodes}. Skipping.")
                    continue
                
                # Project onto POD basis: Basis.T @ Solution
                # (k, nodes) @ (nodes, timesteps) = (k, timesteps)
                projected = pod_basis.T @ solution  # Shape: (num_modes, timesteps)
                
                # Save with same filename as original
                output_file = output_dir / sim_file.name
                np.save(output_file, projected)
                num_saved += 1
                
                if num_saved % 5 == 0:
                    print(f"    Processed {num_saved}/{len(sim_files)} simulations...")
            
            print(f"  ✓ {data_type.upper()}: {num_saved} simulations projected and saved")
            print(f"    Output directory: {output_dir}")
            print(f"    Projected shape per simulation: ({num_modes}, {solution.shape[1]})")
    
    print("\n" + "=" * 80)
    print("PROJECTION COMPLETE!")
    print("=" * 80)
    print("\nOutput directory structure:")
    for comp in components:
        for data_type in data_types:
            output_dir = linear_projected_dir / f'{data_type}_u{comp}'
            if output_dir.exists():
                files = list(output_dir.glob('*.npy'))
                if files:
                    # Load first file to show shape
                    first_file = natsort.natsorted(files)[0]
                    data = np.load(first_file)
                    print(f"  {output_dir}/: {len(files)} files, shape {data.shape}")
    print()

if __name__ == '__main__':
    project_solutions()

