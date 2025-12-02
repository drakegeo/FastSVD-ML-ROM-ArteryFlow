"""
Load and prepare data for LSTM training.

This module loads encoded data from CAE, parameters from simulation_parameters.json,
and creates input/output sequences for LSTM training.
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
    # The encoded data might be from training split only, so we calculate from actual size
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
    
    # Note: The encoded data might contain fewer simulations than the full training set
    # if CAE used a train/val split. We'll use all available parameters and let
    # the load_encoded_data function determine the actual number of simulations.
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


def create_lstm_sequences(encoded_data, parameters, time_window, time_frames, latent_dim):
    """
    Create input/output sequences for LSTM training.
    
    For each simulation, creates sequences by sliding window from time_window to time_frames.
    Each input sequence contains time_window timesteps of latent vectors + parameter value.
    
    Parameters:
    -----------
    encoded_data : np.ndarray
        Encoded data with shape (num_simulations, time_frames, latent_dim)
    parameters : np.ndarray
        Normalized parameters with shape (num_simulations, 1)
    time_window : int
        Time window size (number of timesteps in input sequence)
    time_frames : int
        Maximum timestep to use (typically time_sol)
    latent_dim : int
        Latent space dimension
    
    Returns:
    --------
    tuple : (input_sequences, output_sequences)
        input_sequences: np.ndarray with shape (num_samples, time_window, latent_dim + 1)
        output_sequences: np.ndarray with shape (num_samples, latent_dim)
    """
    num_simulations = encoded_data.shape[0]
    
    # Calculate total number of sequences
    # If time_window == time_frames, we create 1 sequence per simulation
    # (using all timesteps except last to predict last, or using all to predict next if available)
    num_sequences = 0
    for snapshot in range(num_simulations):
        if time_window >= time_frames:
            # If window is >= total timesteps, create 1 sequence using all available timesteps
            num_sequences += 1
        else:
            # Otherwise, create sequences by sliding window
            num_sequences += max(0, time_frames - time_window)
    
    # Initialize arrays
    input_sequences = np.zeros((num_sequences, time_window, latent_dim + 1))
    output_sequences = np.zeros((num_sequences, latent_dim))
    
    sample_idx = 0
    
    for snapshot in range(num_simulations):
        lstm_snapshot = encoded_data[snapshot, :, :]  # Shape: (time_frames, latent_dim)
        param_value = parameters[snapshot, 0]  # Scalar parameter value
        
        if time_window >= time_frames:
            # Use all but the last timestep as input, predict the last timestep
            # Input: timesteps [0:time_frames-1] (all but last)
            available_timesteps = time_frames - 1
            input_sequences[sample_idx, :available_timesteps, :latent_dim] = lstm_snapshot[:available_timesteps, :]
            input_sequences[sample_idx, :available_timesteps, latent_dim] = param_value
            # Pad remaining timesteps with last available timestep if needed
            if time_window > available_timesteps:
                input_sequences[sample_idx, available_timesteps:, :latent_dim] = lstm_snapshot[available_timesteps - 1:available_timesteps, :]
                input_sequences[sample_idx, available_timesteps:, latent_dim] = param_value
            # Output: last timestep
            output_sequences[sample_idx, :] = lstm_snapshot[time_frames - 1, :]
            sample_idx += 1
        else:
            # Create sequences by sliding window
            for t in range(time_window, time_frames):
                # Input: time_window timesteps of latent vectors
                input_sequences[sample_idx, :, :latent_dim] = lstm_snapshot[t - time_window:t, :]
                
                # Add parameter value to all timesteps in the window
                input_sequences[sample_idx, :, latent_dim] = param_value
                
                # Output: next timestep's latent vector
                output_sequences[sample_idx, :] = lstm_snapshot[t, :]
                
                sample_idx += 1
    
    return input_sequences, output_sequences


def prepare_lstm_data(encoded_file_path=None, params_file_path=None, val_split=0.1, random_seed=42):
    """
    Complete pipeline: load encoded data, load and normalize parameters, create sequences.
    
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
    # Load encoded data (already reshaped to (num_simulations, time_sol, latent_dim))
    encoded_data = load_encoded_data(encoded_file_path)
    
    # Get actual number of simulations from encoded data shape
    # encoded_data shape is (num_simulations, time_sol, latent_dim)
    actual_num_sims = encoded_data.shape[0]  # This is already the number of simulations
    
    # Load and normalize parameters
    normalized_params, norm_params = load_and_normalize_parameters(params_file_path)
    
    # Match parameters to actual number of simulations in encoded data
    if normalized_params.shape[0] != actual_num_sims:
        print(f"Warning: Parameters count ({normalized_params.shape[0]}) doesn't match "
              f"encoded data simulations ({actual_num_sims}). Using first {actual_num_sims} parameters.")
        normalized_params = normalized_params[:actual_num_sims]
    
    # Get config parameters
    time_sol = config['time_sol']
    latent_dim = config['latent_CAE_2D']
    
    # Create sequences
    # Use time_sol as the time window size for LSTM input sequences
    # Since we have exactly time_sol timesteps, we use time_sol-1 as window to predict the last timestep
    # This creates 1 sequence per simulation (input: [0:time_sol-1], output: [time_sol-1])
    # OR if we want multiple sequences, we could use a smaller window, but user specified time_sol
    # For now, using time_sol as window means we predict the last available timestep
    time_window = time_sol  # Use time_sol as the sequence length
    
    input_sequences, output_sequences = create_lstm_sequences(
        encoded_data, normalized_params, time_window, time_sol, latent_dim
    )
    
    # Shuffle and split
    if val_split > 0:
        np.random.seed(random_seed)
        num_samples = input_sequences.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        split_idx = int(num_samples * (1 - val_split))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_input = input_sequences[train_indices]
        train_output = output_sequences[train_indices]
        val_input = input_sequences[val_indices]
        val_output = output_sequences[val_indices]
    else:
        train_input = input_sequences
        train_output = output_sequences
        val_input = None
        val_output = None
    
    return train_input, train_output, val_input, val_output, norm_params

