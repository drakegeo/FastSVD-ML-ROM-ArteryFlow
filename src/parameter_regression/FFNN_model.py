"""
Feed-Forward Neural Network (FFNN) Model.

This module defines the FFNN architecture for mapping time and parameter values
to latent space vectors.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import config


class FFNNPredictor(nn.Module):
    """
    Feed-Forward Neural Network for predicting latent vectors from time and parameter.
    
    Architecture:
    - Input: 2 (time, parameter)
    - 3 hidden layers with 32 units each, ReLU activation
    - Output: latent_dim (4)
    """
    def __init__(self, input_size=2, hidden_size=32, num_layers=3, latent_dim=None):
        """
        Initialize FFNN model.
        
        Parameters:
        -----------
        input_size : int
            Input feature size (default: 2 for [time, parameter])
        hidden_size : int
            Hidden size for each layer (default: 32)
        num_layers : int
            Number of hidden layers (default: 3)
        latent_dim : int, optional
            Output latent dimension (default: from config)
        """
        super(FFNNPredictor, self).__init__()
        
        if latent_dim is None:
            latent_dim = config['latent_CAE_2D']
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Build layers
        layers = []
        
        # First layer: input -> hidden
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer: hidden -> latent_dim
        layers.append(nn.Linear(hidden_size, latent_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch, 2) where 2 = [time, parameter]
        
        Returns:
        --------
        torch.Tensor : Output tensor with shape (batch, latent_dim)
        """
        return self.model(x)


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


def build_ffnn_model(input_size=2, hidden_size=32, num_layers=3, latent_dim=None, device=None):
    """
    Build FFNN model.
    
    Parameters:
    -----------
    input_size : int
        Input feature size (default: 2)
    hidden_size : int
        Hidden size for each layer (default: 32)
    num_layers : int
        Number of hidden layers (default: 3)
    latent_dim : int, optional
        Latent space dimension (default: from config)
    device : torch.device, optional
        Device to place model on (default: auto-detect)
    
    Returns:
    --------
    tuple : (model, device)
    """
    if latent_dim is None:
        latent_dim = config['latent_CAE_2D']
    
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    
    # Create model
    model = FFNNPredictor(input_size, hidden_size, num_layers, latent_dim)
    model = model.to(device)
    
    return model, device
