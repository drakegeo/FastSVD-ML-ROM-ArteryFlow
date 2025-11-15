"""
Preprocessing script to extract velocity components from .dat files
and prepare them for SVD processing.

This script processes all training and test datasets, extracts x, y, z components,
and creates metadata with input parameters.
"""
import sys
from pathlib import Path
import numpy as np
import os
import natsort
import json
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.extract_velocity_components import extract_components_from_dat

def extract_parameter_from_folder(folder_name):
    """
    Extract parameter value from folder name.
    Examples: "1_0.07_train" -> 0.07, "10_0.5_test" -> 0.5
    
    Parameters:
    -----------
    folder_name : str
        Folder name (e.g., "5_0.39_train")
    
    Returns:
    --------
    float : Parameter value, or None if not found
    """
    # Pattern: number_parameter_train or number_parameter_test
    match = re.search(r'_([0-9]+\.[0-9]+)_(train|test)', folder_name)
    if match:
        return float(match.group(1))
    return None

def process_dataset(dataset_dir, output_base_dir, sim_index, data_type, data_format='concatenated'):
    """
    Process a single dataset folder and extract all components.
    Combines all timesteps into one file per component.
    
    Parameters:
    -----------
    dataset_dir : Path
        Path to dataset folder (e.g., data/HFM/1_0.07_train)
    output_base_dir : Path
        Base directory for output
    sim_index : int
        Simulation index for naming output files
    data_type : str
        'train' or 'test'
    data_format : str
        Format of data in .dat files
    """
    dataset_dir = Path(dataset_dir)
    output_base_dir = Path(output_base_dir)
    
    # Create preprocessed directory structure
    preprocessed_dir = output_base_dir / 'data' / 'preprocessed'
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output directories based on data type (train or test)
    output_dirs = {
        'x': preprocessed_dir / f'{data_type}_ux',
        'y': preprocessed_dir / f'{data_type}_uy',
        'z': preprocessed_dir / f'{data_type}_uz'
    }
    
    for comp_dir in output_dirs.values():
        comp_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all .dat files
    dat_files = natsort.natsorted([f for f in os.listdir(dataset_dir) if f.endswith('.dat')])
    
    print(f"Processing {len(dat_files)} timesteps from {dataset_dir.name}")
    
    # First, get dimensions from first valid file
    num_nodes = None
    num_timesteps = len(dat_files)
    
    # Find first valid file to get dimensions
    for dat_file in dat_files:
        dat_path = dataset_dir / dat_file
        try:
            data = np.loadtxt(dat_path)
            num_nodes = len(data) // 3
            break
        except:
            continue
    
    if num_nodes is None:
        print(f"  ✗ No valid files found in {dataset_dir.name}")
        return None
    
    print(f"  Found {num_timesteps} timesteps, {num_nodes} nodes per timestep")
    
    # Pre-allocate arrays for each component: shape (nodes, timesteps)
    stacked_components = {
        'x': np.zeros((num_nodes, num_timesteps), dtype=np.float32),
        'y': np.zeros((num_nodes, num_timesteps), dtype=np.float32),
        'z': np.zeros((num_nodes, num_timesteps), dtype=np.float32)
    }
    
    # Process all timesteps and fill pre-allocated arrays
    valid_idx = 0
    for i, dat_file in enumerate(dat_files):
        if (i + 1) % 20 == 0:
            print(f"  Loading timestep {i+1}/{num_timesteps}...")
        
        dat_path = dataset_dir / dat_file
        
        # Extract components from this timestep
        try:
            components = extract_components_from_dat(dat_path, None, data_format)
            
            # Store each component directly in pre-allocated array
            for comp in ['x', 'y', 'z']:
                stacked_components[comp][:, valid_idx] = components[comp].astype(np.float32)
            valid_idx += 1
        except Exception as e:
            print(f"  Error processing {dat_file}: {e}")
            continue
    
    # Trim arrays to actual number of valid timesteps
    if valid_idx < num_timesteps:
        print(f"  Warning: Only {valid_idx} valid timesteps out of {num_timesteps}")
        for comp in ['x', 'y', 'z']:
            stacked_components[comp] = stacked_components[comp][:, :valid_idx]
    
    # Save each component
    for comp in ['x', 'y', 'z']:
        print(f"  Component {comp}: shape {stacked_components[comp].shape} (nodes x timesteps)")
        
        # Save as one file per simulation
        output_name = f"sim_u{sim_index:04d}.npy"
        output_path = output_dirs[comp] / output_name
        np.save(output_path, stacked_components[comp])
    
    print(f"  ✓ Completed {dataset_dir.name} -> sim_u{sim_index:04d}.npy")
    
    # Return metadata
    parameter = extract_parameter_from_folder(dataset_dir.name)
    return {
        'simulation_index': sim_index,
        'folder_name': dataset_dir.name,
        'parameter': parameter,
        'data_type': data_type,
        'num_timesteps': valid_idx,
        'num_nodes': num_nodes
    }

def preprocess_all_data(data_dir='data/HFM', output_dir='.', data_format='concatenated'):
    """
    Preprocess all training and test datasets.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing datasets
    output_dir : str
        Base directory for output
    data_format : str
        Format of data in .dat files ('concatenated' or 'interleaved')
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Ensure preprocessed directory exists
    preprocessed_dir = output_dir / 'data' / 'preprocessed'
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata directory
    metadata_dir = preprocessed_dir / 'inputs'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {preprocessed_dir}")
    print(f"Metadata directory: {metadata_dir}")
    
    # Find all dataset folders (both train and test)
    all_folders = [d for d in data_dir.iterdir() if d.is_dir()]
    train_folders = [d for d in all_folders if 'train' in d.name]
    test_folders = [d for d in all_folders if 'test' in d.name]
    
    print("=" * 60)
    print("Preprocessing Data")
    print("=" * 60)
    print(f"Found {len(train_folders)} training datasets")
    print(f"Found {len(test_folders)} test datasets")
    print(f"Data format: {data_format}")
    print()
    
    all_metadata = {'train': [], 'test': []}
    
    # Process training datasets
    if train_folders:
        print("\n" + "=" * 60)
        print("Processing Training Data")
        print("=" * 60)
        for sim_idx, folder in enumerate(sorted(train_folders), start=1):
            # Check if already processed
            output_name = f"sim_u{sim_idx:04d}.npy"
            already_processed = all(
                (preprocessed_dir / f'train_u{comp}' / output_name).exists()
                for comp in ['x', 'y', 'z']
            )
            
            if already_processed:
                print(f"\nSkipping: {folder.name} (simulation {sim_idx}) - already processed")
                # Still extract metadata
                parameter = extract_parameter_from_folder(folder.name)
                all_metadata['train'].append({
                    'simulation_index': sim_idx,
                    'folder_name': folder.name,
                    'parameter': parameter
                })
                continue
            
            print(f"\nProcessing: {folder.name} (simulation {sim_idx})")
            metadata = process_dataset(folder, output_dir, sim_idx, 'train', data_format)
            if metadata:
                all_metadata['train'].append(metadata)
    
    # Process test datasets
    if test_folders:
        print("\n" + "=" * 60)
        print("Processing Test Data")
        print("=" * 60)
        for sim_idx, folder in enumerate(sorted(test_folders), start=1):
            # Check if already processed
            output_name = f"sim_u{sim_idx:04d}.npy"
            already_processed = all(
                (preprocessed_dir / f'test_u{comp}' / output_name).exists()
                for comp in ['x', 'y', 'z']
            )
            
            if already_processed:
                print(f"\nSkipping: {folder.name} (simulation {sim_idx}) - already processed")
                # Still extract metadata
                parameter = extract_parameter_from_folder(folder.name)
                all_metadata['test'].append({
                    'simulation_index': sim_idx,
                    'folder_name': folder.name,
                    'parameter': parameter
                })
                continue
            
            print(f"\nProcessing: {folder.name} (simulation {sim_idx})")
            metadata = process_dataset(folder, output_dir, sim_idx, 'test', data_format)
            if metadata:
                all_metadata['test'].append(metadata)
    
    # Save metadata
    metadata_file = metadata_dir / 'simulation_parameters.json'
    with open(metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_file}")
    
    # Also save as separate files for easy access
    for data_type in ['train', 'test']:
        if all_metadata[data_type]:
            params_file = metadata_dir / f'{data_type}_parameters.txt'
            with open(params_file, 'w') as f:
                f.write(f"{data_type.upper()} Simulation Parameters\n")
                f.write("=" * 60 + "\n")
                for meta in all_metadata[data_type]:
                    f.write(f"sim_u{meta['simulation_index']:04d}.npy: ")
                    f.write(f"parameter = {meta['parameter']}, ")
                    f.write(f"folder = {meta['folder_name']}\n")
            print(f"Parameters saved to: {params_file}")
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print("\nCreated directories:")
    print("  - data/preprocessed/train_ux/, train_uy/, train_uz/")
    print("  - data/preprocessed/test_ux/, test_uy/, test_uz/")
    print("  - data/preprocessed/inputs/ (metadata)")
    print("\nYou can now run: python src/linear_projection/process_all_components.py")

if __name__ == '__main__':
    # Simple usage: just run the script with defaults
    preprocess_all_data()
