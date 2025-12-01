"""
CAE-2D Training Script

This script trains a 2D Convolutional Autoencoder for nonlinear dimensionality
reduction of POD-projected velocity field data.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import config
from src.nonlinear_reduction.CAE_model import build_cae_2d, get_device
from src.nonlinear_reduction.load_data import load_and_prepare_cae_data


def standardize_data(data, mean=None, std=None):
    """
    Standardize data using mean and standard deviation.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data to standardize
    mean : np.ndarray, optional
        Mean for standardization. If None, computed from data.
    std : np.ndarray, optional
        Standard deviation for standardization. If None, computed from data.
    
    Returns:
    --------
    tuple : (standardized_data, mean, std)
    """
    if mean is None:
        mean = np.mean(data, axis=0, keepdims=True)
    if std is None:
        std = np.std(data, axis=0, keepdims=True)
        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)
    
    standardized = (data - mean) / std
    return standardized, mean, std


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Parameters:
    -----------
    model : nn.Module
        PyTorch model
    dataloader : DataLoader
        Training data loader
    criterion : nn.Module
        Loss function
    optimizer : optim.Optimizer
        Optimizer
    device : torch.device
        Device to use
    
    Returns:
    --------
    float : Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_data in dataloader:
        inputs = batch_data[0].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, inputs)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Parameters:
    -----------
    model : nn.Module
        PyTorch model
    dataloader : DataLoader
        Validation data loader
    criterion : nn.Module
        Loss function
    device : torch.device
        Device to use
    
    Returns:
    --------
    float : Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in dataloader:
            inputs = batch_data[0].to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, inputs)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    """Main training function."""
    # Get device
    device = get_device()
    
    # Get parameters from config
    lr = config['lr_CAE_2D']
    batch_size = config['batch_CAE_2D']
    epochs = config['epochs_CAE_2D']
    latent_dim = config['latent_CAE_2D']
    val_split = config.get('val_split_CAE_2D', 0.1)
    spatial_shape = config.get('spatial_shape_CAE_2D', (16, 16))
    
    print("=" * 60)
    print("CAE-2D Training")
    print("=" * 60)
    print(f"Learning rate: {lr}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Validation split: {val_split}")
    print(f"Spatial shape: {spatial_shape}")
    print("=" * 60)
    
    # Load and prepare training data
    print("\nLoading training data...")
    train_data, val_data = load_and_prepare_cae_data(
        data_type='train',
        spatial_shape=spatial_shape,
        val_split=val_split,
        random_seed=42
    )
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    
    # Standardize data
    print("\nStandardizing data...")
    train_data_std, train_mean, train_std = standardize_data(train_data)
    val_data_std = (val_data - train_mean) / train_std
    
    # Define output directory (nonlinear_reduction/output folder)
    nonlinear_reduction_dir = Path(__file__).parent
    output_dir = nonlinear_reduction_dir / 'output'
    
    # Save standardization parameters
    scaling_dir = output_dir / 'scaling_data'
    scaling_dir.mkdir(parents=True, exist_ok=True)
    np.save(scaling_dir / 'stdmean_CAE2D.npy', {'mean': train_mean, 'std': train_std})
    print(f"Saved standardization parameters to {scaling_dir / 'stdmean_CAE2D.npy'}")
    
    # Convert to PyTorch tensors and reshape for Conv2D: (N, H, W, C) -> (N, C, H, W)
    train_tensor = torch.FloatTensor(train_data_std).permute(0, 3, 1, 2)
    val_tensor = torch.FloatTensor(val_data_std).permute(0, 3, 1, 2)
    
    # Create data loaders
    train_dataset = TensorDataset(train_tensor, train_tensor)
    val_dataset = TensorDataset(val_tensor, val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Build model
    print("\nBuilding CAE-2D model...")
    model, encoder, decoder, device = build_cae_2d(latent_dim=latent_dim, device=device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 50
    
    # Create output directories in nonlinear_reduction/output folder
    weights_dir = output_dir / 'DL_weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / 'DL_data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting training...")
    print("-" * 60)
    
    # Training loop
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save model weights
            torch.save(model.state_dict(), weights_dir / 'weights_CAE2D.pth')
            torch.save(encoder.state_dict(), weights_dir / 'enc_CAE2D.pth')
            torch.save(decoder.state_dict(), weights_dir / 'dec_CAE2D.pth')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print("-" * 60)
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    
    # Load best model
    model.load_state_dict(torch.load(weights_dir / 'weights_CAE2D.pth'))
    encoder.load_state_dict(torch.load(weights_dir / 'enc_CAE2D.pth'))
    decoder.load_state_dict(torch.load(weights_dir / 'dec_CAE2D.pth'))
    
    # Generate encoded and decoded data for all training data
    print("\nGenerating encoded and decoded data...")
    model.eval()
    with torch.no_grad():
        # Process in batches to avoid memory issues
        all_encoded = []
        all_decoded = []
        
        for batch_data in train_loader:
            inputs = batch_data[0].to(device)
            encoded = encoder(inputs)
            decoded = decoder(encoded)
            
            all_encoded.append(encoded.cpu().numpy())
            all_decoded.append(decoded.cpu().numpy())
        
        encoded_data = np.concatenate(all_encoded, axis=0)
        decoded_data = np.concatenate(all_decoded, axis=0)
    
    # Save encoded and decoded data
    np.save(data_dir / 'CAE2D_enc.npy', encoded_data)
    np.save(data_dir / 'CAE2D_dec.npy', decoded_data)
    print(f"Saved encoded data: {data_dir / 'CAE2D_enc.npy'}")
    print(f"Saved decoded data: {data_dir / 'CAE2D_dec.npy'}")
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'epochs': list(range(1, len(train_losses) + 1))
    }
    
    import json
    results_dir = output_dir / 'results_csv'
    results_dir.mkdir(parents=True, exist_ok=True)
    history_file = results_dir / 'CAE_2D.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history: {history_file}")
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
