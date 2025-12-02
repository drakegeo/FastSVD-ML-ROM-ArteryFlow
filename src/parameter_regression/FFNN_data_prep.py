"""
Load and prepare data for FFNN training.

This module loads encoded data from CAE, parameters from simulation_parameters.json,
and creates input/output pairs for FFNN training.
"""
import sys
from pathlib import Path

import numpy as np
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import config


def load_encoded_data(encoded_file_path=None):
    """
    Load encoded data from CAE and reshape to (num_simulations, time_sol, latent_dim).
    
    Parameters:
    -----------
    encoded_file_path : Path, optional
        Path to encoded data file. Default: from nonlinear_reduction/output/DL_data/CAE2D_enc.npy
    
    Returns:
    --------
    np.ndarray : Reshaped encoded data with shape (num_simulations, time_sol, latent_dim)
    """
    if encoded_file_path is None:
        nonlinear_reduction_dir = project_root / 'src' / 'nonlinear_reduction' / 'output' / 'DL_data'
        encoded_file_path = nonlinear_reduction_dir / 'CAE2D_enc.npy'
    
    if not encoded_file_path.exists():
        raise FileNotFoundError(f"Encoded data file not found: {encoded_file_path}")
    
    # Load encoded data
    encoded_data = np.load(encoded_file_path)  # Shape: (total_samples, latent_dim)
    
    # Get parameters
    time_sol = config['time_sol']
    latent_dim = config['latent_CAE_2D']
    
    # Calculate number of simulations from actual data size
    total_samples = encoded_data.shape[0]
    num_simulations = total_samples // time_sol
    
    if total_samples % time_sol != 0:
        raise ValueError(
            f"Encoded data size ({total_samples}) is not divisible by time_sol ({time_sol}). "
            f"This suggests the data might not be properly organized."
        )
    
    print(f"Loaded encoded data: {total_samples} samples = {num_simulations} simulations × {time_sol} timesteps")
    
    # Reshape to (num_simulations, time_sol, latent_dim)
    reshaped_data = encoded_data.reshape(num_simulations, time_sol, latent_dim)
    
    return reshaped_data


def load_and_normalize_parameters(params_file_path=None):
    """
    Load parameters from simulation_parameters.json and normalize using min-max.
    
    Parameters:
    -----------
    params_file_path : Path, optional
        Path to parameters file. Default: from data/preprocessed/inputs/simulation_parameters.json
    
    Returns:
    --------
    tuple : (normalized_parameters, normalization_params)
        normalized_parameters: np.ndarray with shape (num_simulations, 1)
        normalization_params: dict with 'min' and 'max' keys
    """
    if params_file_path is None:
        params_file_path = project_root / 'data' / 'preprocessed' / 'inputs' / 'simulation_parameters.json'
    
    if not params_file_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_file_path}")
    
    with open(params_file_path, 'r') as f:
        params_data = json.load(f)
    
    # Extract training parameters
    train_params = [sim['parameter'] for sim in params_data['train']]
    params_array = np.array(train_params).reshape(-1, 1)  # Shape: (num_simulations, 1)
    
    # Min-max normalization
    param_min = params_array.min()
    param_max = params_array.max()
    
    if param_max == param_min:
        # Avoid division by zero
        normalized_params = np.zeros_like(params_array)
    else:
        normalized_params = (params_array - param_min) / (param_max - param_min)
    
    normalization_params = {
        'min': param_min,
        'max': param_max
    }
    
    return normalized_params, normalization_params


def prepare_ffnn_data(encoded_data, parameters, time_window):
    """
    Prepare input/output pairs for FFNN training.
    
    For each simulation and each timestep in [0, time_window), creates:
    - Input: [normalized_time, normalized_parameter]
    - Output: [latent_vector at that timestep]
    
    Parameters:
    -----------
    encoded_data : np.ndarray
        Encoded data with shape (num_simulations, time_sol, latent_dim)
    parameters : np.ndarray
        Normalized parameters with shape (num_simulations, 1)
    time_window : int
        Number of timesteps to use (first time_window timesteps)
    
    Returns:
    --------
    tuple : (inputs, outputs)
        inputs: np.ndarray with shape (num_samples, 2) where 2 = [time, parameter]
        outputs: np.ndarray with shape (num_samples, latent_dim)
    """
    num_simulations = encoded_data.shape[0]
    latent_dim = encoded_data.shape[2]
    
    # Normalize time to [0, 1] range
    # Time values: 0, 1, 2, ..., time_window-1
    # Normalized: 0, 1/(time_window-1), 2/(time_window-1), ..., 1
    if time_window > 1:
        time_values = np.linspace(0, 1, time_window).reshape(-1, 1)
    else:
        time_values = np.array([[0.0]])
    
    # Create input/output pairs
    num_samples = num_simulations * time_window
    inputs = np.zeros((num_samples, 2))  # [time, parameter]
    outputs = np.zeros((num_samples, latent_dim))
    
    sample_idx = 0
    for sim_idx in range(num_simulations):
        param_value = parameters[sim_idx, 0]  # Scalar parameter value
        
        for t in range(time_window):
            # Input: [normalized_time, normalized_parameter]
            inputs[sample_idx, 0] = time_values[t, 0]
            inputs[sample_idx, 1] = param_value
            
            # Output: latent vector at this timestep
            outputs[sample_idx, :] = encoded_data[sim_idx, t, :]
            
            sample_idx += 1
    
    return inputs, outputs


def prepare_ffnn_training_data(encoded_file_path=None, params_file_path=None, val_split=0.1, random_seed=42):
    """
    Complete pipeline: load encoded data, load and normalize parameters, prepare FFNN data.
    
    Parameters:
    -----------
    encoded_file_path : Path, optional
        Path to encoded data file
    params_file_path : Path, optional
        Path to parameters file
    val_split : float
        Validation split ratio (default: 0.1)
    random_seed : int
        Random seed for shuffling (default: 42)
    
    Returns:
    --------
    tuple : (train_input, train_output, val_input, val_output, normalization_params)
        All arrays are numpy arrays
        normalization_params: dict with parameter normalization info
    """
    # Load encoded data
    encoded_data = load_encoded_data(encoded_file_path)
    
    # Get actual number of simulations from encoded data shape
    actual_num_sims = encoded_data.shape[0]
    
    # Load and normalize parameters
    normalized_params, norm_params = load_and_normalize_parameters(params_file_path)
    
    # Match parameters to actual number of simulations in encoded data
    if normalized_params.shape[0] != actual_num_sims:
        print(f"Warning: Parameters count ({normalized_params.shape[0]}) doesn't match "
              f"encoded data simulations ({actual_num_sims}). Using first {actual_num_sims} parameters.")
        normalized_params = normalized_params[:actual_num_sims]
    
    # Get config parameters
    time_window = config['FFNN_time_window']
    
    # Prepare data
    inputs, outputs = prepare_ffnn_data(encoded_data, normalized_params, time_window)
    
    # Shuffle and split
    if val_split > 0:
        np.random.seed(random_seed)
        num_samples = inputs.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        split_idx = int(num_samples * (1 - val_split))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_input = inputs[train_indices]
        train_output = outputs[train_indices]
        val_input = inputs[val_indices]
        val_output = outputs[val_indices]
    else:
        train_input = inputs
        train_output = outputs
        val_input = None
        val_output = None
    
    return train_input, train_output, val_input, val_output, norm_params

