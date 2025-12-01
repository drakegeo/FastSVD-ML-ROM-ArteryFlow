# FastSVD-ML-ROM: Machine Learning Platform for Reduced Order Models

## Overview

Digital twins have emerged as a key technology for optimizing the performance of engineering products and systems. High-fidelity numerical simulations constitute the backbone of engineering design, providing insight into the performance of complex systems. However, large-scale, dynamic, non-linear models require significant computational resources and are prohibitive for real-time digital twin applications. 

To this end, reduced order models (ROMs) are employed to approximate the high-fidelity solutions while accurately capturing the dominant aspects of the physical behavior. The present repository proposes a new machine learning (ML) platform for the development of ROMs to handle large-scale numerical problems dealing with transient nonlinear partial differential equations.

## Features

FastSVD-ML-ROM is a comprehensive framework that combines multiple machine learning techniques for efficient reduced order modeling:

- **SVD Update Methodology**: Computes a linear subspace of multi-fidelity solutions during the simulation process
- **Convolutional Autoencoders**: Enables nonlinear dimensionality reduction
- **Feed-Forward Neural Networks**: Maps input parameters to latent spaces
- **Long-Short Term Memory (LSTM) Networks**: Predicts and forecasts the dynamics of parametric solutions

## Machine Learning Framework

The FastSVD-ML-ROM framework utilizes a multi-stage approach:

1. **SVD-based Linear Subspace**: Computes and updates a linear subspace representation of multi-fidelity solutions during simulation
2. **Nonlinear Dimensionality Reduction**: Convolutional autoencoders compress high-dimensional solution spaces into compact latent representations
3. **Parameter Mapping**: Feed-forward neural networks learn the relationship between input parameters and latent space coordinates
4. **Temporal Dynamics**: LSTM networks capture and predict the time evolution of parametric solutions

## Data

**Note:** The training and test data for this project are not included in the repository. To obtain the data, please contact the repository maintainer. Once you receive the data, place it in the `data/` directory following the structure described below.

The framework has been demonstrated on 3D blood flow simulations inside an arterial segment. The `data/` directory should contain high-fidelity model (HFM) data organized as follows:

### Training Data
- **10 parameterized training datasets**: Each folder (e.g., `1_0.07_train`, `2_0.15_train`, etc.) contains 160 `.dat` files representing solution snapshots at different time steps
- Training datasets cover various parameter configurations for learning the parameter-to-latent-space mapping

### Test Data
- **2 test datasets**: `10_0.5_test` and `11_0.42_test`, each containing 200 `.dat` files
- Test datasets are used for validation and performance evaluation on unseen parameter configurations

### Data Format
- All data files are in `.dat` format containing high-fidelity simulation snapshots
- The folder naming convention indicates different parameter sets (e.g., `1_0.07` represents parameter set 1 with value 0.07)
- These HFM snapshots are used to train the convolutional autoencoders, feed-forward neural networks, and LSTM networks

## Environment

### Requirements

#### 1. Check CUDA Version

First, verify your CUDA version to ensure compatibility:

```bash
nvidia-smi
```

This will display your CUDA version and GPU information. The framework has been tested with CUDA 11.8.

#### 2. Install PyTorch with CUDA Support

Install PyTorch and torchvision with CUDA 11.8 support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
```

The `--no-cache-dir` flag is recommended to avoid memory issues during installation.

**Verify PyTorch CUDA installation:**

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

#### 3. Install Other Dependencies

Install the remaining required dependencies:

```bash
pip install -r requirements.txt
```

### Setup

1. Clone the repository
2. Check your CUDA version using `nvidia-smi`
3. Install PyTorch with CUDA support (see above)
4. Install dependencies from `requirements.txt`
5. **Request the training and test data from the repository maintainer and place it in the `data/` directory**
6. Configure the model parameters as needed in `src/config.py`

## Usage

### 1. Linear Projection (SVD)

First, perform SVD-based linear projection on your high-fidelity simulation data:

```bash
python src/linear_projection/process_all_components.py
```

This generates projected data in `data/linear_projected/` organized by component (ux, uy, uz) and data type (train/test).

### 2. CAE-2D Training

Train the 2D Convolutional Autoencoder for nonlinear dimensionality reduction:

```bash
python src/nonlinear_reduction/CAE_main.py
```

**Configuration:**
- Edit `src/config.py` to adjust training parameters:
  - `lr_CAE_2D`: Learning rate (default: 0.0005)
  - `batch_CAE_2D`: Batch size (default: 20)
  - `epochs_CAE_2D`: Number of epochs (default: 2000)
  - `latent_CAE_2D`: Latent space dimension (default: 4)
  - `val_split_CAE_2D`: Validation split ratio (default: 0.1)

**Outputs:**
The training script saves the following in `src/nonlinear_reduction/output/`:
- `DL_weights/weights_CAE2D.pth`: Full model weights
- `DL_weights/enc_CAE2D.pth`: Encoder weights
- `DL_weights/dec_CAE2D.pth`: Decoder weights
- `DL_data/CAE2D_enc.npy`: Encoded latent representations
- `DL_data/CAE2D_dec.npy`: Decoded reconstructions
- `scaling_data/stdmean_CAE2D.npy`: Data standardization parameters
- `results_csv/CAE_2D.json`: Training history (losses per epoch)

**Features:**
- Automatic CUDA/CPU device detection
- Early stopping (patience: 50 epochs)
- Model checkpointing (saves best model based on validation loss)
- Random train/validation split with reproducible shuffling

## Results

The efficiency of the FastSVD-ML-ROM framework has been demonstrated for 3D blood flow inside an arterial segment. The accuracy of the reconstructed results indicates the robustness of the proposed approach.

## Citation

If you use this work in your research, please cite the following paper:

```
Drakoulas, G.I., Gortsas, T.V., Bourantas, G.C., Burganos, V.N., Polyzos, D. (2023). 
FastSVD-ML–ROM: A reduced-order modeling framework based on machine learning for real-time applications. 
Computer Methods in Applied Mechanics and Engineering.
```

Paper available at: [https://www.sciencedirect.com/science/article/abs/pii/S0045782523002797](https://www.sciencedirect.com/science/article/abs/pii/S0045782523002797)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
