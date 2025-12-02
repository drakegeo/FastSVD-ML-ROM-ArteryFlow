"""
LSTM Model for Time Series Prediction.

This module defines the LSTM architecture for predicting next timestep's latent vector
from a sequence of previous latent vectors and parameter values.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import config


class LSTMPredictor(nn.Module):
    """
    LSTM model for predicting next timestep's latent vector.
    
    Architecture:
    - 4 LSTM layers (80 units each)
    - First 3 LSTM layers: return_sequences=True
    - Last LSTM layer: return_sequences=False
    - Final Dense layer: outputs latent_dim
    """
    def __init__(self, input_size, hidden_size=80, num_layers=4, latent_dim=None):
        """
        Initialize LSTM model.
        
        Parameters:
        -----------
        input_size : int
            Input feature size (latent_dim + 1, where +1 is parameter)
        hidden_size : int
            Hidden size for LSTM layers (default: 80)
        num_layers : int
            Number of LSTM layers (default: 4)
        latent_dim : int, optional
            Output latent dimension (default: from config)
        """
        super(LSTMPredictor, self).__init__()
        
        if latent_dim is None:
            latent_dim = config['latent_CAE_2D']
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        # In PyTorch, LSTM always returns sequences, we just take what we need
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm4 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # Final dense layer
        self.dense = nn.Linear(hidden_size, latent_dim)
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch, sequence_length, input_size)
        
        Returns:
        --------
        torch.Tensor : Output tensor with shape (batch, latent_dim)
        """
        # LSTM layers
        # First 3 layers: use all sequence outputs
        x, _ = self.lstm1(x)  # (batch, seq_len, hidden_size)
        x, _ = self.lstm2(x)  # (batch, seq_len, hidden_size)
        x, _ = self.lstm3(x)  # (batch, seq_len, hidden_size)
        
        # Last LSTM layer: take only the last timestep output
        x, _ = self.lstm4(x)  # (batch, seq_len, hidden_size)
        x = x[:, -1, :]  # Take last timestep: (batch, hidden_size)
        
        # Final dense layer
        x = self.dense(x)  # (batch, latent_dim)
        
        return x


def get_device():
    """
    Get the appropriate device (CUDA if available, else CPU).
    
    Returns:
    --------
    torch.device : Device to use for computation
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    return device


def build_lstm_model(latent_dim=None, hidden_size=80, num_layers=4, device=None):
    """
    Build LSTM model.
    
    Parameters:
    -----------
    latent_dim : int, optional
        Latent space dimension (default: from config)
    hidden_size : int
        Hidden size for LSTM layers (default: 80)
    num_layers : int
        Number of LSTM layers (default: 4)
    device : torch.device, optional
        Device to place model on (default: auto-detect)
    
    Returns:
    --------
    tuple : (model, device)
    """
    if latent_dim is None:
        latent_dim = config['latent_CAE_2D']
    
    # Input size is latent_dim + 1 (parameter)
    input_size = latent_dim + 1
    
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    
    # Create model
    model = LSTMPredictor(input_size, hidden_size, num_layers, latent_dim)
    model = model.to(device)
    
    return model, device
