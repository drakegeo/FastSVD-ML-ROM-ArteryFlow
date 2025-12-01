"""
Convolutional Autoencoder (CAE) 2D model definition using PyTorch.

This module defines the encoder-decoder architecture for non-linear
dimensionality reduction of POD-projected data.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import config
from src.config import config

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(beta * x)
    """
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta
    
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Encoder(nn.Module):
    """
    Encoder network for CAE-2D.
    
    Architecture:
    Conv2D(30) -> MaxPool -> Conv2D(20) -> MaxPool -> Conv2D(10) -> MaxPool
    -> Flatten -> Dense(40) -> Dense(10) -> Dense(latent_dim)
    """
    def __init__(self, latent_dim, activation='swish'):
        super(Encoder, self).__init__()
        
        # Choose activation function
        if activation == 'swish':
            self.activ = Swish()
        elif activation == 'elu':
            self.activ = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}. Use 'swish' or 'elu'")
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 30, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(30, 20, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv2d(20, 10, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        # After 3 max pools: 16x16 -> 8x8 -> 4x4 -> 2x2
        # So flattened size is 2 * 2 * 10 = 40
        self.fc1 = nn.Linear(2 * 2 * 10, 40)
        self.fc2 = nn.Linear(40, 10)
        self.fc3 = nn.Linear(10, latent_dim)
    
    def forward(self, x):
        # Input: (batch, 3, 16, 16)
        x = self.activ(self.conv1(x))  # (batch, 30, 16, 16)
        x = self.pool1(x)  # (batch, 30, 8, 8)
        
        x = self.activ(self.conv2(x))  # (batch, 20, 8, 8)
        x = self.pool2(x)  # (batch, 20, 4, 4)
        
        x = self.activ(self.conv3(x))  # (batch, 10, 4, 4)
        x = self.pool3(x)  # (batch, 10, 2, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 40)
        
        # Fully connected layers
        x = self.activ(self.fc1(x))  # (batch, 40)
        x = self.activ(self.fc2(x))  # (batch, 10)
        x = self.fc3(x)  # (batch, latent_dim)
        
        return x


class Decoder(nn.Module):
    """
    Decoder network for CAE-2D.
    
    Architecture:
    Dense(10) -> Dense(40) -> Dense(2*2*3) -> Reshape(2,2,3)
    -> Conv2D(10) -> Upsample -> Conv2D(20) -> Upsample -> Conv2D(30) -> Upsample
    -> Conv2D(3)
    """
    def __init__(self, latent_dim, activation='swish'):
        super(Decoder, self).__init__()
        
        # Choose activation function
        if activation == 'swish':
            self.activ = Swish()
        elif activation == 'elu':
            self.activ = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}. Use 'swish' or 'elu'")
        
        # Fully connected layers
        self.fc1 = nn.Linear(latent_dim, 10)
        self.fc2 = nn.Linear(10, 40)
        self.fc3 = nn.Linear(40, 2 * 2 * 3)  # Reshape to (2, 2, 3)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv4 = nn.Conv2d(30, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Input: (batch, latent_dim)
        x = self.activ(self.fc1(x))  # (batch, 10)
        x = self.activ(self.fc2(x))  # (batch, 40)
        x = self.activ(self.fc3(x))  # (batch, 12)
        
        # Reshape to (batch, 3, 2, 2)
        x = x.view(x.size(0), 3, 2, 2)
        
        # Convolutional layers with upsampling
        x = self.activ(self.conv1(x))  # (batch, 10, 2, 2)
        x = self.upsample1(x)  # (batch, 10, 4, 4)
        
        x = self.activ(self.conv2(x))  # (batch, 20, 4, 4)
        x = self.upsample2(x)  # (batch, 20, 8, 8)
        
        x = self.activ(self.conv3(x))  # (batch, 30, 8, 8)
        x = self.upsample3(x)  # (batch, 30, 16, 16)
        
        x = self.conv4(x)  # (batch, 3, 16, 16) - linear activation (no activation)
        
        return x


class CAE2D(nn.Module):
    """
    Full Convolutional Autoencoder model.
    """
    def __init__(self, latent_dim, activation='swish'):
        super(CAE2D, self).__init__()
        self.encoder = Encoder(latent_dim, activation)
        self.decoder = Decoder(latent_dim, activation)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


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


def build_cae_2d(latent_dim=None, activation='swish', device=None):
    """
    Build 2D Convolutional Autoencoder model.
    
    Parameters:
    -----------
    latent_dim : int, optional
        Latent space dimension (default: from config)
    activation : str
        Activation function, 'swish' or 'elu' (default: 'swish')
    device : torch.device, optional
        Device to place model on (default: auto-detect using get_device())
    
    Returns:
    --------
    tuple : (model, encoder, decoder, device)
    """
    if latent_dim is None:
        latent_dim = config['latent_CAE_2D']
    
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    
    # Create model components
    encoder = Encoder(latent_dim, activation)
    decoder = Decoder(latent_dim, activation)
    model = CAE2D(latent_dim, activation)
    
    # Move to device
    model = model.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    return model, encoder, decoder, device
