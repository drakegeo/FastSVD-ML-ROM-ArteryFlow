"""
Load and prepare linear projected data for CAE training.

This module loads the projected POD data from data/linear_projected/,
reshapes it to 16x16 spatial format, and prepares it for 3-kernel CAE input.
"""
import sys
from pathlib import Path

import numpy as np
from natsort import natsorted

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import config


def load_projected_data(data_type='train', components=None):
    """
    Load projected POD data for specified components and data type.
    
    Parameters:
    -----------
    data_type : str
        'train' or 'test'
    components : list, optional
        List of components to load, e.g., ['x', 'y', 'z']. Default: ['x', 'y', 'z']
    
    Returns:
    --------
    dict : Dictionary with keys as components, values as lists of arrays
           Each array has shape (modes, timesteps) = (256, 160)
    """
    if components is None:
        components = ['x', 'y', 'z']
    
    project_root = Path(__file__).parent.parent.parent
    linear_projected_dir = project_root / 'data' / 'linear_projected'
    
    data = {}
    for comp in components:
        comp_dir = linear_projected_dir / f'{data_type}_u{comp}'
        
        if not comp_dir.exists():
            raise FileNotFoundError(f"Directory not found: {comp_dir}")
        
        # Load all simulation files for this component
        sim_files = natsorted([f for f in comp_dir.glob('*.npy')])
        
        if not sim_files:
            raise ValueError(f"No simulation files found in {comp_dir}")
        
        # Load all simulations for this component
        component_data = []
        for sim_file in sim_files:
            sim_data = np.load(sim_file)  # Shape: (256, 160)
            component_data.append(sim_data)
        
        data[comp] = component_data
    
    return data


def combine_simulations(component_data):
    """
    Combine multiple simulations into a single array.
    
    For each simulation (256, 160), concatenates along timestep axis.
    Result: (256, total_timesteps) where total_timesteps = num_simulations * 160
    
    Parameters:
    -----------
    component_data : list
        List of arrays, each with shape (256, 160)
    
    Returns:
    --------
    np.ndarray : Combined array with shape (256, num_simulations * 160)
    """
    if not component_data:
        raise ValueError("Empty component_data list")
    
    # Concatenate along timestep axis (axis=1)
    combined = component_data[0]  # Shape: (256, 160)
    for sim_data in component_data[1:]:
        combined = np.append(combined, sim_data, axis=1)
    
    return combined  # Shape: (256, total_timesteps)


def reshape_to_spatial(data_combined, spatial_shape=(16, 16)):
    """
    Reshape 1D mode data to 2D spatial format.
    
    Parameters:
    -----------
    data_combined : np.ndarray
        Combined data with shape (256, timesteps)
    spatial_shape : tuple
        Target spatial shape, default (16, 16) since 16*16 = 256
    
    Returns:
    --------
    np.ndarray : Reshaped data with shape (timesteps, 16, 16)
    """
    num_modes, num_timesteps = data_combined.shape
    
    if num_modes != spatial_shape[0] * spatial_shape[1]:
        raise ValueError(
            f"Number of modes ({num_modes}) must equal "
            f"spatial_shape[0] * spatial_shape[1] ({spatial_shape[0] * spatial_shape[1]})"
        )
    
    # Reshape: (256, timesteps) -> (timesteps, 16, 16)
    # Transpose first to get (timesteps, 256), then reshape
    reshaped = data_combined.T.reshape(num_timesteps, spatial_shape[0], spatial_shape[1])
    
    return reshaped  # Shape: (timesteps, 16, 16)


def prepare_cae_input(data_dict, spatial_shape=(16, 16)):
    """
    Prepare data for 3-kernel CAE input.
    
    Loads, combines, reshapes, and stacks x, y, z components.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with keys 'x', 'y', 'z', each containing list of arrays
    spatial_shape : tuple
        Spatial shape for reshaping, default (16, 16)
    
    Returns:
    --------
    np.ndarray : Final CAE input with shape (timesteps, 16, 16, 3)
    """
    reshaped_data = {}
    
    for comp in ['x', 'y', 'z']:
        if comp not in data_dict:
            raise ValueError(f"Missing component '{comp}' in data_dict")
        
        # Combine all simulations for this component
        combined = combine_simulations(data_dict[comp])  # (256, total_timesteps)
        
        # Reshape to spatial format
        reshaped = reshape_to_spatial(combined, spatial_shape)  # (total_timesteps, 16, 16)
        
        reshaped_data[comp] = reshaped
    
    # Check all components have same number of timesteps
    timesteps = [data.shape[0] for data in reshaped_data.values()]
    if len(set(timesteps)) != 1:
        raise ValueError(f"Inconsistent timesteps across components: {timesteps}")
    
    # Stack components: (timesteps, 16, 16, 3)
    final_input = np.stack([
        reshaped_data['x'],
        reshaped_data['y'],
        reshaped_data['z']
    ], axis=-1)
    
    return final_input  # Shape: (timesteps, 16, 16, 3)


def load_and_prepare_cae_data(data_type='train', spatial_shape=None, val_split=None, random_seed=42):
    """
    Complete pipeline: load, combine, reshape, and prepare CAE input.
    Optionally splits training data into train/validation sets with random shuffle.
    
    Parameters:
    -----------
    data_type : str
        'train' or 'test'
    spatial_shape : tuple, optional
        Spatial shape for reshaping. Default: from config['spatial_shape_CAE_2D']
    val_split : float, optional
        Validation split ratio (0.0 to 1.0). Default: from config['val_split_CAE_2D']
        Only used when data_type='train'
    random_seed : int
        Random seed for shuffling. Default: 42
    
    Returns:
    --------
    np.ndarray or tuple : 
        - If data_type='test': CAE input with shape (timesteps, 16, 16, 3)
        - If data_type='train' and val_split > 0: (train_data, val_data) tuple
        - If data_type='train' and val_split = 0: CAE input with shape (timesteps, 16, 16, 3)
    """
    if spatial_shape is None:
        spatial_shape = config.get('spatial_shape_CAE_2D', (16, 16))
    
    # Load projected data
    data_dict = load_projected_data(data_type=data_type, components=['x', 'y', 'z'])
    
    # Prepare for CAE
    cae_input = prepare_cae_input(data_dict, spatial_shape=spatial_shape)
    
    # For training data, optionally split into train/val
    if data_type == 'train':
        if val_split is None:
            val_split = config.get('val_split_CAE_2D', 0.1)
        
        if val_split > 0:
            # Set random seed for reproducibility
            np.random.seed(random_seed)
            
            # Shuffle the data
            num_samples = cae_input.shape[0]
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            
            # Split indices
            split_idx = int(num_samples * (1 - val_split))
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            
            # Split data
            train_data = cae_input[train_indices]
            val_data = cae_input[val_indices]
            
            return train_data, val_data
        else:
            # No validation split, return all data
            return cae_input
    
    return cae_input

