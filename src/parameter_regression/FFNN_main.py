"""
FFNN Training Script

This script trains a Feed-Forward Neural Network to map time and parameter values
to latent space vectors.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import config
from src.parameter_regression.FFNN_model import build_ffnn_model, get_device
from src.parameter_regression.FFNN_data_prep import prepare_ffnn_training_data


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
        targets = batch_data[1].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, targets)
        
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
            targets = batch_data[1].to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    """Main training function."""
    # Get device
    device = get_device()
    
    # Get parameters from config
    lr = config['lr_FFNN']
    batch_size = config['batch_FFNN']
    epochs = config['epochs_FFNN']
    latent_dim = config['latent_CAE_2D']
    time_window = config['FFNN_time_window']
    val_split = config.get('val_split_FFNN', 0.1)
    hidden_size = 32
    num_layers = 3
    
    print("=" * 60)
    print("FFNN Training")
    print("=" * 60)
    print(f"Learning rate: {lr}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Time window: {time_window}")
    print(f"Hidden size: {hidden_size}")
    print(f"Number of layers: {num_layers}")
    print(f"Validation split: {val_split}")
    print("=" * 60)
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    train_input, train_output, val_input, val_output, norm_params = prepare_ffnn_training_data(
        val_split=val_split,
        random_seed=42
    )
    
    print(f"Training input shape: {train_input.shape}")
    print(f"Training output shape: {train_output.shape}")
    if val_input is not None:
        print(f"Validation input shape: {val_input.shape}")
        print(f"Validation output shape: {val_output.shape}")
    
    # Convert to PyTorch tensors
    train_input_tensor = torch.FloatTensor(train_input)
    train_output_tensor = torch.FloatTensor(train_output)
    
    # Create data loaders
    train_dataset = TensorDataset(train_input_tensor, train_output_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if val_input is not None:
        val_input_tensor = torch.FloatTensor(val_input)
        val_output_tensor = torch.FloatTensor(val_output)
        val_dataset = TensorDataset(val_input_tensor, val_output_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    
    # Build model
    print("\nBuilding FFNN model...")
    model, device = build_ffnn_model(
        input_size=2,
        hidden_size=hidden_size,
        num_layers=num_layers,
        latent_dim=latent_dim,
        device=device
    )
    
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
    
    # Create output directories
    parameter_regression_dir = Path(__file__).parent
    output_dir = parameter_regression_dir / 'output'
    weights_dir = output_dir / 'DL_weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    scaling_dir = output_dir / 'scaling_data'
    scaling_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / 'results_csv'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save normalization parameters
    np.save(scaling_dir / 'param_norm_FFNN.npy', norm_params)
    print(f"Saved parameter normalization to {scaling_dir / 'param_norm_FFNN.npy'}")
    
    print("\nStarting training...")
    print("-" * 60)
    
    # Training loop
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
        else:
            val_loss = float('inf')
            val_losses.append(None)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            if val_loader is not None:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}")
        
        # Save best model
        if val_loader is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save model weights
            torch.save(model.state_dict(), weights_dir / 'weights_FFNN.pth')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience and val_loader is not None:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print("-" * 60)
    if val_loader is not None:
        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    else:
        print("Training completed.")
    
    # Load best model if validation was used
    if val_loader is not None:
        model.load_state_dict(torch.load(weights_dir / 'weights_FFNN.pth', weights_only=True))
    
    # Save final model
    torch.save(model.state_dict(), weights_dir / 'weights_FFNN_final.pth')
    print(f"Saved model weights to {weights_dir / 'weights_FFNN.pth'}")
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses if val_loader is not None else None,
        'epochs': list(range(1, len(train_losses) + 1))
    }
    
    history_file = results_dir / 'FFNN.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history: {history_file}")
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
