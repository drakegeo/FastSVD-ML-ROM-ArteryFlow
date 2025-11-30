"""
Project solutions onto POD basis matrices for all components.

For each simulation file (nodes × 160 timesteps), projects it onto the POD basis
to get compressed representation (number_of_modes × 160).

Result: Each simulation is compressed from (nodes, 160) to (k, 160) where k is
the number of POD modes in the basis.

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
        
        # Process train and test data
        projected_data = {'train': [], 'test': []}
        
        for data_type in data_types:
            input_dir = preprocessed_dir / f'{data_type}_u{comp}'
            
            if not input_dir.exists():
                print(f"  Warning: Input directory not found: {input_dir}")
                continue
            
            # Get all simulation files
            sim_files = natsort.natsorted([f for f in input_dir.glob('*.npy')])
            
            if not sim_files:
                print(f"  No simulation files found in {input_dir}")
                continue
            
            print(f"  Processing {data_type.upper()} data: {len(sim_files)} simulations")
            
            for sim_file in sim_files:
                # Load solution: shape (nodes, 160)
                solution = np.load(sim_file)  # Shape: (nodes, timesteps)
                
                if solution.shape[0] != num_nodes:
                    print(f"    Warning: {sim_file.name} has {solution.shape[0]} nodes, "
                          f"expected {num_nodes}. Skipping.")
                    continue
                
                # Project onto POD basis: Basis.T @ Solution
                # (k, nodes) @ (nodes, timesteps) = (k, timesteps)
                projected = pod_basis.T @ solution  # Shape: (num_modes, timesteps)
                
                projected_data[data_type].append(projected)
                
                if len(projected_data[data_type]) % 5 == 0:
                    print(f"    Processed {len(projected_data[data_type])}/{len(sim_files)} simulations...")
            
            print(f"  ✓ {data_type.upper()}: {len(projected_data[data_type])} simulations projected")
            print(f"    Projected shape per simulation: ({num_modes}, {solution.shape[1]})")
        
        # Save projected data
        if projected_data['train'] or projected_data['test']:
            output_file = linear_projected_dir / f'vel_snapshot_{comp}.npy'
            
            # Save as list of arrays (one per simulation)
            # Each array is shape (num_modes, timesteps)
            # Note: Train has 160 timesteps, test has 200 timesteps - different shapes are expected
            all_projected = []
            if projected_data['train']:
                all_projected.extend(projected_data['train'])
            if projected_data['test']:
                all_projected.extend(projected_data['test'])
            
            # Save as object array to handle different shapes (train: 160, test: 200 timesteps)
            # Create object array properly to handle different shapes
            obj_array = np.empty(len(all_projected), dtype=object)
            for i, arr in enumerate(all_projected):
                obj_array[i] = arr
            np.save(output_file, obj_array, allow_pickle=True)
            print(f"\n  ✓ Saved projected data: {output_file}")
            print(f"    Total simulations: {len(all_projected)}")
            if all_projected:
                train_shape = f"({num_modes}, 160)" if projected_data['train'] else "N/A"
                test_shape = f"({num_modes}, 200)" if projected_data['test'] else "N/A"
                print(f"    Train shape: {train_shape}, Test shape: {test_shape}")
    
    print("\n" + "=" * 80)
    print("PROJECTION COMPLETE!")
    print("=" * 80)
    print("\nOutput files:")
    for comp in components:
        output_file = linear_projected_dir / f'vel_snapshot_{comp}.npy'
        if output_file.exists():
            data = np.load(output_file, allow_pickle=True)
            if len(data) > 0:
                # Show first train and test shapes if available
                first_shape = data[0].shape
                if len(data) > 10:  # If we have both train and test
                    last_shape = data[-1].shape
                    print(f"  {output_file.name}: {len(data)} simulations")
                    print(f"    Train: shape {first_shape}, Test: shape {last_shape}")
                else:
                    print(f"  {output_file.name}: {len(data)} simulations, "
                          f"shape {first_shape}")
    print()

if __name__ == '__main__':
    project_solutions()

