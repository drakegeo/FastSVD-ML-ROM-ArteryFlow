import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import natsort
from pathlib import Path

# Handle imports - support both relative and absolute imports
try:
    from ..config import config
except ImportError:
    # If relative import fails, use absolute import
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config import config

def load(file, filepath):
    """
    Load simulation data from .npy file.
    Each file contains all timesteps: shape (nodes, timesteps)
    Returns the array as-is.
    """
    snapshot = np.load(os.path.join(filepath, file))
    return snapshot

# SVD of the Matrix B
def svd(B_re, error, Full):
    U, S, VT = np.linalg.svd(B_re, full_matrices=Full)
    k = 1
    for i in range(len(S)):
        if S[i] < S[0] * error:
            k = i + 1
            break
        k = i + 1
    return U, S, VT, k

# truncated SVD of the Matrix B
def trunc(U, S, VT, k):
    Uk = U[:, :k]
    VkT = VT[:k, :]
    Sk = S[:k]
    return Uk, Sk, VkT

# projectors of E
def proj(Uk, E):
    UkE = Uk.T @ E
    UE = Uk @ UkE
    PE = (E - UE)
    return UkE, PE

# QR decomposition of the truncated Matrix
def QR_dec(PE):
    Q, R = np.linalg.qr(PE, mode='reduced')
    return Q, R

# Compose the matrix M
def matrixM(Sk, UkE, s, R):
    k = np.shape(Sk)[0]
    Sk = np.diag(Sk)
    M1 = np.concatenate((Sk, UkE), axis=1)
    M2 = np.concatenate((np.zeros((s, k)), R), axis=1)
    M = np.concatenate((M1, M2), axis=0)  # Size (k+s)*(k+s)
    return M

# Formulate the Left Singular vector
def UA_form(Uk, Q, Um):
    UA1 = np.concatenate((Uk, Q), axis=1)
    UA = UA1 @ Um
    return UA

# Formulate the Right Singular vector
def VA_form(Vk, s, n, V1, Sk):
    k = np.shape(Sk)[0]
    VA1 = np.concatenate((Vk, np.zeros((k, s))), axis=1)
    VA2 = np.concatenate((np.zeros((s, n)), np.eye(s, s)), axis=1)
    VA = np.concatenate((VA1, VA2), axis=0).T @ V1.T
    return VA

# Reconstruct the Full solution
def recon(B_re, UA, SA, VkT):
    """
    Reconstruct the solution and compute L2 error.
    
    Parameters:
    -----------
    B_re : array, shape (nodes, num_snapshots)
        Original accumulated snapshots
    UA : array, shape (nodes, k)
        Left singular vectors (POD basis)
    SA : array, shape (k,)
        Singular values
    VkT : array, shape (num_snapshots, k)
        Right singular vectors (transposed)
    
    Returns:
    --------
    float : Relative L2 reconstruction error
    """
    # Reconstruction: AR = UA @ diag(SA) @ VkT.T
    # UA: (nodes, k), diag(SA): (k, k), VkT.T: (k, num_snapshots)
    AR = UA @ np.diag(SA) @ VkT.T
    # Removed plt.show() - it blocks execution and is not needed
    L2_norm_error = np.linalg.norm(B_re - AR, ord=2) / np.linalg.norm(B_re, ord=2)
    return L2_norm_error

def process_component(component):
    """
    Process SVD update for a given velocity component using training data only.
    
    Parameters:
    -----------
    component : str
        Component to process ('x', 'y', or 'z')
    
    Returns:
    --------
    dict : Dictionary with processing results
    """
    # Normalize component to lowercase
    component = component.lower()
    
    if component not in ['x', 'y', 'z']:
        raise ValueError(f"Component must be 'x', 'y', or 'z'. Got: {component}")
    
    # Get truncation error for the selected component
    trunc_err = config[f'trunc_error_{component}']
    
    # Paths - input data directory (training data only from data/preprocessed/)
    input_dir = f'./data/preprocessed/train_u{component}/'
    
    # Output directories in data folder
    data_dir = Path('./data')
    pod_basis_dir = data_dir / 'POD_basis'
    linear_projected_dir = data_dir / 'linear_projected'
    svd_results_dir = data_dir / 'SVD_results'
    
    # Create output directories if they don't exist
    pod_basis_dir.mkdir(parents=True, exist_ok=True)
    linear_projected_dir.mkdir(parents=True, exist_ok=True)
    svd_results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing component: {component.upper()}")
    print(f"Truncation error: {trunc_err}")
    print(f"Input directory: {input_dir}")
    
    trun_size = []
    trunc_l2n_err = []
    i = 0

    # Check if input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    name_sim = []
    for file in os.listdir(input_dir):
        name_sim.append(file)
    sim_short_ = natsort.natsorted(name_sim)
    print(f"Found {len(sim_short_)} simulation files to process")
    
    # Process each simulation file (each contains all timesteps)
    for file in sim_short_:
        snap = load(file, input_dir)  # Shape: (nodes, timesteps)
        print(f'Loaded {file}, shape: {snap.shape} (nodes x timesteps)')
        
        # Process each timestep column as a snapshot
        num_timesteps = snap.shape[1]
        for t_idx in range(num_timesteps):
            # Extract one timestep: shape (nodes,)
            timestep_snapshot = snap[:, t_idx]
            
            if i == 0:
                # First snapshot: initialize B
                B = timestep_snapshot.reshape(-1, 1)  # Shape: (nodes, 1)
                print(f'First snapshot loaded, shape: {B.shape}')
                U, S, VT, k = svd(B, trunc_err, False)
                Uk, Sk, VkT = trunc(U, S, VT, k)
                i += 1
                print(f'Initial truncation size: {k}')
                l = k
                fom_tru = Uk @ np.diag(Sk) @ VkT
                l2_norm_er = np.linalg.norm(B - fom_tru, ord=2) / np.linalg.norm(B, ord=2)
                print(f'Initial L2 norm error: {l2_norm_er}')

            else:
                i += 1
                if (i - 1) % 20 == 0:
                    print(f'Processing snapshot {i} (file: {file}, timestep: {t_idx+1}/{num_timesteps})')
                
                # New snapshot as column vector
                E = timestep_snapshot.reshape(-1, 1)  # Shape: (nodes, 1)
                s = np.shape(E)[1]  # Should be 1
                n = np.shape(B)[1]
                m = np.shape(B)[0]

                UkE, PE = proj(Uk, E)
                Q, R = QR_dec(PE)

                M = matrixM(Sk, UkE, s, R)
                Um, SA, VmT, km = svd(M, trunc_err, False)
                Um, SA, VmT = trunc(Um, SA, VmT, km)

                UA = UA_form(Uk, Q, Um)
                VA = VA_form(VkT, s, n, VmT, Sk)
                B = np.concatenate((B, E), axis=1)

                Uk, Sk, VkT = UA, SA, VA.T

                # calculate truncation
                l = 1
                for j in range(len(Sk)):
                    if Sk[j] < Sk[0] * trunc_err:
                        l = j + 1
                        break
                    l = j + 1

                if i == 40:
                    l = 256

                Uk, Sk, VkT = trunc(Uk, Sk, VkT, l)
                
                # Only compute expensive reconstruction error every 20 snapshots
                # This is the main bottleneck - full matrix reconstruction is O(nodes × snapshots)
                if (i - 1) % 20 == 0:
                    l2_norm_er = recon(B, Uk, Sk, VkT)
                    print(f'  L2 error: {l2_norm_er:.6e}, Truncation size: {l}')
                else:
                    l2_norm_er = 0.0  # Placeholder - not computed to save time
            
            trun_size.append(l)
            trunc_l2n_err.append(l2_norm_er)

    # Save POD basis
    pod_basis_file = pod_basis_dir / f'u{component}.npy'
    np.save(pod_basis_file, Uk)
    print(f'POD basis saved to: {pod_basis_file}')
    print(f'POD basis shape: {np.shape(Uk)}')

    # Save SVD results
    trunc_size_file = svd_results_dir / f'trunc_size_{component}.npy'
    trunc_l2_file = svd_results_dir / f'trunc_l2_{component}.npy'
    np.save(trunc_size_file, trun_size)
    np.save(trunc_l2_file, trunc_l2n_err)
    print(f'SVD results saved to: {svd_results_dir}')

    # Project all snapshots and save
    print(f'Projecting all snapshots...')
    name_sim = []
    for file in os.listdir(input_dir):
        name_sim.append(file)
    sim_short_ = natsort.natsorted(name_sim)
    fnl_snapshot = []

    for file in sim_short_:
        print(f'Projecting: {file}')
        snapshot = np.load(os.path.join(input_dir, file))  # Shape: (nodes, timesteps)
        # Project all timesteps: Uk.T @ snapshot -> (basis_size, timesteps)
        projected = Uk.T @ snapshot  # Shape: (basis_size, timesteps)
        fnl_snapshot.append(projected)

    print(f'Projected snapshots shape: {np.shape(fnl_snapshot)}')
    
    # Save linear projected data
    linear_projected_file = linear_projected_dir / f'vel_snapshot_{component}.npy'
    np.save(linear_projected_file, fnl_snapshot)
    print(f'Linear projected data saved to: {linear_projected_file}')
    print('Processing complete!')
    
    return {
        'component': component,
        'pod_basis': Uk,
        'trunc_size': trun_size,
        'trunc_l2_err': trunc_l2n_err,
        'projected_snapshots': fnl_snapshot
    }

# This module contains functions for SVD processing.
# To process components, use process_all_components.py as the main script.
# 
# Functions available:
#   - process_component(component): Process a single velocity component
#   - load, svd, trunc, proj, QR_dec, matrixM, UA_form, VA_form, recon: Helper functions

