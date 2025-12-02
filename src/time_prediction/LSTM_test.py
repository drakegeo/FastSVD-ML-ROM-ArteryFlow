"""
LSTM Test/Evaluation Script

This script evaluates the trained LSTM model on test data.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import config
from src.time_prediction.LSTM_model import LSTMPredictor, get_device
from src.time_prediction.LSTM_data_prep import load_encoded_data, load_and_normalize_parameters


def load_test_encoded_data():
    """
    Load encoded test data from CAE.
    
    Returns:
    --------
    np.ndarray : Test encoded data with shape (num_test_simulations, time_sol, latent_dim)
    """
    nonlinear_reduction_dir = project_root / 'src' / 'nonlinear_reduction' / 'output' / 'DL_data'
    encoded_file = nonlinear_reduction_dir / 'CAE2D_enc_test.npy'
    
    if not encoded_file.exists():
        raise FileNotFoundError(
            f"Test encoded data file not found: {encoded_file}\n"
            "You need to run CAE_main.py on test data first to generate this file."
        )
    
    encoded_data = np.load(encoded_file)  # Shape: (total_samples, latent_dim)
    
    time_sol = config['time_sol']
    latent_dim = config['latent_CAE_2D']
    
    # Calculate number of test simulations
    total_samples = encoded_data.shape[0]
    num_simulations = total_samples // time_sol
    
    if total_samples % time_sol != 0:
        raise ValueError(
            f"Test encoded data size ({total_samples}) is not divisible by time_sol ({time_sol})."
        )
    
    print(f"Loaded test encoded data: {total_samples} samples = {num_simulations} simulations × {time_sol} timesteps")
    
    # Reshape to (num_simulations, time_sol, latent_dim)
    reshaped_data = encoded_data.reshape(num_simulations, time_sol, latent_dim)
    
    return reshaped_data


def load_test_parameters():
    """
    Load and normalize test parameters.
    
    Returns:
    --------
    tuple : (normalized_parameters, normalization_params)
    """
    params_file = project_root / 'data' / 'preprocessed' / 'inputs' / 'simulation_parameters.json'
    
    if not params_file.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_file}")
    
    with open(params_file, 'r') as f:
        params_data = json.load(f)
    
    # Extract test parameters
    test_params = [sim['parameter'] for sim in params_data['test']]
    params_array = np.array(test_params).reshape(-1, 1)
    
    # Load normalization parameters from training
    time_prediction_dir = Path(__file__).parent
    scaling_dir = time_prediction_dir / 'output' / 'scaling_data'
    norm_file = scaling_dir / 'param_norm_LSTM.npy'
    
    if not norm_file.exists():
        raise FileNotFoundError(
            f"Parameter normalization file not found: {norm_file}\n"
            "You need to train the LSTM model first."
        )
    
    norm_params = np.load(norm_file, allow_pickle=True).item()
    param_min = norm_params['min']
    param_max = norm_params['max']
    
    # Normalize test parameters using training normalization
    if param_max == param_min:
        normalized_params = np.zeros_like(params_array)
    else:
        normalized_params = (params_array - param_min) / (param_max - param_min)
    
    return normalized_params, norm_params


def create_test_sequences(encoded_data, parameters, time_window, latent_dim):
    """
    Create test sequences for evaluation.
    
    For each simulation, creates input sequences using the time window.
    
    Parameters:
    -----------
    encoded_data : np.ndarray
        Encoded data with shape (num_simulations, time_sol, latent_dim)
    parameters : np.ndarray
        Normalized parameters with shape (num_simulations, 1)
    time_window : int
        Time window size (should be time_sol)
    latent_dim : int
        Latent space dimension
    
    Returns:
    --------
    tuple : (input_sequences, true_outputs)
        input_sequences: np.ndarray with shape (num_simulations, time_window, latent_dim + 1)
        true_outputs: np.ndarray with shape (num_simulations, latent_dim)
    """
    num_simulations = encoded_data.shape[0]
    time_sol = encoded_data.shape[1]
    
    # Create sequences - one per simulation
    # Use all but last timestep as input, predict last timestep
    input_sequences = np.zeros((num_simulations, time_window, latent_dim + 1))
    true_outputs = np.zeros((num_simulations, latent_dim))
    
    for snapshot in range(num_simulations):
        lstm_snapshot = encoded_data[snapshot, :, :]  # Shape: (time_sol, latent_dim)
        param_value = parameters[snapshot, 0]
        
        if time_window >= time_sol:
            # Use all but last timestep as input
            available_timesteps = time_sol - 1
            input_sequences[snapshot, :available_timesteps, :latent_dim] = lstm_snapshot[:available_timesteps, :]
            input_sequences[snapshot, :available_timesteps, latent_dim] = param_value
            # Pad if needed
            if time_window > available_timesteps:
                input_sequences[snapshot, available_timesteps:, :latent_dim] = lstm_snapshot[available_timesteps - 1:available_timesteps, :]
                input_sequences[snapshot, available_timesteps:, latent_dim] = param_value
            # True output: last timestep
            true_outputs[snapshot, :] = lstm_snapshot[time_sol - 1, :]
        else:
            # Use time_window timesteps to predict next
            input_sequences[snapshot, :, :latent_dim] = lstm_snapshot[:time_window, :]
            input_sequences[snapshot, :, latent_dim] = param_value
            # True output: timestep after window
            true_outputs[snapshot, :] = lstm_snapshot[time_window, :]
    
    return input_sequences, true_outputs


def evaluate_model(model, input_sequences, true_outputs, device):
    """
    Evaluate model on test data.
    
    Parameters:
    -----------
    model : nn.Module
        Trained LSTM model
    input_sequences : np.ndarray
        Input sequences with shape (num_samples, time_window, latent_dim + 1)
    true_outputs : np.ndarray
        True outputs with shape (num_samples, latent_dim)
    device : torch.device
        Device to use
    
    Returns:
    --------
    dict : Evaluation metrics
    """
    model.eval()
    
    # Convert to tensors
    input_tensor = torch.FloatTensor(input_sequences).to(device)
    true_tensor = torch.FloatTensor(true_outputs).to(device)
    
    # Predict
    with torch.no_grad():
        pred_tensor = model(input_tensor)
    
    # Convert to numpy
    predictions = pred_tensor.cpu().numpy()
    true_values = true_tensor.cpu().numpy()
    
    # Calculate metrics
    mse = np.mean((predictions - true_values) ** 2)
    mae = np.mean(np.abs(predictions - true_values))
    rmse = np.sqrt(mse)
    
    # Per-dimension metrics
    mse_per_dim = np.mean((predictions - true_values) ** 2, axis=0)
    mae_per_dim = np.mean(np.abs(predictions - true_values), axis=0)
    
    # Relative error
    relative_error = np.mean(np.abs(predictions - true_values) / (np.abs(true_values) + 1e-10))
    
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'relative_error': float(relative_error),
        'mse_per_dim': mse_per_dim.tolist(),
        'mae_per_dim': mae_per_dim.tolist(),
        'predictions': predictions.tolist(),
        'true_values': true_values.tolist()
    }
    
    return metrics


def main():
    """Main evaluation function."""
    # Get device
    device = get_device()
    
    # Get parameters from config
    time_sol = config['time_sol']
    latent_dim = config['latent_CAE_2D']
    time_window = config.get('time_window_LSTM', time_sol)
    
    print("=" * 60)
    print("LSTM Test/Evaluation")
    print("=" * 60)
    print(f"Time window: {time_window}")
    print(f"Latent dimension: {latent_dim}")
    print("=" * 60)
    
    # Load test data
    print("\nLoading test data...")
    try:
        test_encoded_data = load_test_encoded_data()
        test_parameters, norm_params = load_test_parameters()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nNote: To test on test data, you need to:")
        print("1. Run CAE_main.py on test data to generate CAE2D_enc_test.npy")
        print("2. Or modify CAE_main.py to also encode test data")
        return
    
    # Match parameters to data
    num_test_sims = test_encoded_data.shape[0]
    if test_parameters.shape[0] != num_test_sims:
        print(f"Warning: Parameters count ({test_parameters.shape[0]}) doesn't match "
              f"test data simulations ({num_test_sims}). Using first {num_test_sims} parameters.")
        test_parameters = test_parameters[:num_test_sims]
    
    # Create test sequences
    print("\nCreating test sequences...")
    test_input, test_output = create_test_sequences(
        test_encoded_data, test_parameters, time_window, latent_dim
    )
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {test_output.shape}")
    
    # Load trained model
    print("\nLoading trained model...")
    time_prediction_dir = Path(__file__).parent
    weights_dir = time_prediction_dir / 'output' / 'DL_weights'
    weights_file = weights_dir / 'weights_LSTM.pth'
    
    if not weights_file.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_file}\n"
            "You need to train the LSTM model first."
        )
    
    # Build model
    input_size = latent_dim + 1
    model = LSTMPredictor(input_size, hidden_size=80, num_layers=4, latent_dim=latent_dim)
    model.load_state_dict(torch.load(weights_file, weights_only=True))
    model = model.to(device)
    
    print("Model loaded successfully.")
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model, test_input, test_output, device)
    
    # Print results
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"MSE:  {metrics['mse']:.6e}")
    print(f"MAE:  {metrics['mae']:.6e}")
    print(f"RMSE: {metrics['rmse']:.6e}")
    print(f"Relative Error: {metrics['relative_error']:.6e}")
    print("\nPer-dimension MSE:")
    for i, mse_dim in enumerate(metrics['mse_per_dim']):
        print(f"  Dimension {i}: {mse_dim:.6e}")
    print("\nPer-dimension MAE:")
    for i, mae_dim in enumerate(metrics['mae_per_dim']):
        print(f"  Dimension {i}: {mae_dim:.6e}")
    print("=" * 60)
    
    # Save results
    results_dir = time_prediction_dir / 'output' / 'results_csv'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / 'LSTM_test_results.json'
    
    # Remove large arrays from saved metrics
    save_metrics = {k: v for k, v in metrics.items() if k not in ['predictions', 'true_values']}
    
    with open(results_file, 'w') as f:
        json.dump(save_metrics, f, indent=2)
    
    print(f"\nSaved test results to: {results_file}")
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()

