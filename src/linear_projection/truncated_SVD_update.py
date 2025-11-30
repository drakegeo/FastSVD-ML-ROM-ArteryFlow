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
    Returns the array as float32 for memory efficiency.
    """
    snapshot = np.load(os.path.join(filepath, file))
    # Convert to float32 to reduce memory usage by 50%
    if snapshot.dtype != np.float32:
        snapshot = snapshot.astype(np.float32)
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
    # Ensure float32 for memory efficiency
    Sk = Sk.astype(np.float32) if Sk.dtype != np.float32 else Sk
    UkE = UkE.astype(np.float32) if UkE.dtype != np.float32 else UkE
    R = R.astype(np.float32) if R.dtype != np.float32 else R
    
    Sk = np.diag(Sk)
    M1 = np.concatenate((Sk, UkE), axis=1)
    M2 = np.concatenate((np.zeros((s, k), dtype=np.float32), R), axis=1)
    M = np.concatenate((M1, M2), axis=0)  # Size (k+s)*(k+s)
    return M

# Formulate the Left Singular vector
def UA_form(Uk, Q, Um):
    # Ensure float32 for memory efficiency
    Uk = Uk.astype(np.float32) if Uk.dtype != np.float32 else Uk
    Q = Q.astype(np.float32) if Q.dtype != np.float32 else Q
    Um = Um.astype(np.float32) if Um.dtype != np.float32 else Um
    
    # Use more memory-efficient matrix multiplication
    # Instead of concatenating first, we can compute: Uk @ Um[:k, :] + Q @ Um[k:, :]
    k = Uk.shape[1]
    UA = Uk @ Um[:k, :] + Q @ Um[k:, :]
    return UA

# Formulate the Right Singular vector
def VA_form(Vk, s, n, V1, Sk):
    k = np.shape(Sk)[0]
    # Ensure float32 for memory efficiency
    Vk = Vk.astype(np.float32) if Vk.dtype != np.float32 else Vk
    V1 = V1.astype(np.float32) if V1.dtype != np.float32 else V1
    
    # Build VA more efficiently by avoiding intermediate large arrays where possible
    VA1 = np.concatenate((Vk, np.zeros((k, s), dtype=np.float32)), axis=1)
    VA2 = np.concatenate((np.zeros((s, n), dtype=np.float32), np.eye(s, s, dtype=np.float32)), axis=1)
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
    VkT : array, shape (k, num_snapshots) 
        Right singular vectors (note: this is actually VA, not VkT, matching old script)
    
    Returns:
    --------
    float : Relative L2 reconstruction error
    """
    # Reconstruction: AR = UA @ diag(SA) @ VkT
    # Matching old script exactly: uses VkT directly (line 142 in old script)
    # UA: (nodes, k), diag(SA): (k, k), VkT: (k, num_snapshots)
    # Optimize: use broadcasting instead of creating full diagonal matrix
    # AR = UA @ (SA[:, None] * VkT) is equivalent but more memory-efficient
    AR = UA @ (SA[:, None] * VkT)
    # Removed plt.show() - it blocks execution and is not needed
    # Use Frobenius norm instead of spectral norm (ord=2) to avoid MemoryError with large matrices
    # Frobenius norm is more memory-efficient and still a valid reconstruction error metric
    L2_norm_error = np.linalg.norm(B_re - AR, ord='fro') / np.linalg.norm(B_re, ord='fro')
    return L2_norm_error

def process_component(component):
    """
    Process SVD update for a given velocity component using training data only.
    Calculates and saves the POD basis for the component.
    Linear projection of data is done separately in another script.
    
    Parameters:
    -----------
    component : str
        Component to process ('x', 'y', or 'z')
    
    Returns:
    --------
    dict : Dictionary with processing results containing:
        - component: Component name
        - pod_basis: POD basis matrix
        - trunc_size: List of truncation sizes for each snapshot
        - trunc_l2_err: List of L2 reconstruction errors for each snapshot
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
    svd_results_dir = data_dir / 'SVD_results'
    
    # Create output directories if they don't exist
    pod_basis_dir.mkdir(parents=True, exist_ok=True)
    svd_results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing component: {component.upper()}")
    print(f"Truncation error: {trunc_err}")
    print(f"Input directory: {input_dir}")
    
    trun_size = []
    trunc_l2n_err = []
    i = 0
    last_l2_error = None  # Track last computed error to carry forward

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
                # First snapshot: initialize B (use float32 for memory efficiency)
                B = timestep_snapshot.reshape(-1, 1).astype(np.float32)  # Shape: (nodes, 1)
                print(f'First snapshot loaded, shape: {B.shape}')
                U, S, VT, k = svd(B, trunc_err, False)
                Uk, Sk, VkT = trunc(U, S, VT, k)
                i += 1
                print(f'Initial truncation size: {k}')
                l = k
                # Use broadcasting for memory efficiency
                fom_tru = Uk @ (Sk[:, None] * VkT)
                # Use Frobenius norm to avoid MemoryError (spectral norm requires SVD which is memory-intensive)
                l2_norm_er = np.linalg.norm(B - fom_tru, ord='fro') / np.linalg.norm(B, ord='fro')
                last_l2_error = l2_norm_er  # Store initial error
                print(f'Initial L2 norm error: {l2_norm_er}')

            else:
                i += 1
                if (i - 1) % 20 == 0:
                    print(f'Processing snapshot {i} (file: {file}, timestep: {t_idx+1}/{num_timesteps})')
                
                # New snapshot as column vector (ensure float32)
                E = timestep_snapshot.reshape(-1, 1).astype(np.float32)  # Shape: (nodes, 1)
                s = np.shape(E)[1]  # Should be 1
                n = np.shape(B)[1]
                m = np.shape(B)[0]

                UkE, PE = proj(Uk, E)
                Q, R = QR_dec(PE)

                M = matrixM(Sk, UkE, s, R)
                Um, SA_full, VmT, km = svd(M, trunc_err, False)
                # Save full SA before truncation for truncation size calculation
                Um, SA, VmT = trunc(Um, SA_full, VmT, km)

                UA = UA_form(Uk, Q, Um)
                VA = VA_form(VkT, s, n, VmT, Sk)
                # Ensure B stays as float32 when concatenating
                B = np.concatenate((B, E), axis=1).astype(np.float32)

                Uk, Sk, VkT = UA, SA, VA.T

                # calculate truncation using full SA (before truncation)
                # Note: km is already the truncation size from SA_full, but we recalculate
                # to match the original script's logic (which uses Sk after truncation)
                l = 1
                for j in range(len(SA_full)):
                    if SA_full[j] < SA_full[0] * trunc_err:
                        l = j + 1
                        break
                    l = j + 1
                # l should equal km in most cases, but we recalculate for consistency

                Uk, Sk, VkT = trunc(Uk, Sk, VkT, l)
                
                # Only compute expensive reconstruction error every 20 snapshots
                # This is the main bottleneck - full matrix reconstruction is O(nodes × snapshots)
                if (i - 1) % 20 == 0:
                    l2_norm_er = recon(B, Uk, Sk, VkT)
                    last_l2_error = l2_norm_er  # Update last computed error
                    print(f'  L2 error: {l2_norm_er:.6e}, Truncation size: {l}')
                else:
                    # Use last computed error value (carry forward) instead of 0
                    l2_norm_er = last_l2_error
            
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
    
    print('Processing complete!')
    
    return {
        'component': component,
        'pod_basis': Uk,
        'trunc_size': trun_size,
        'trunc_l2_err': trunc_l2n_err
    }

# This module contains functions for SVD processing.
# To process components, use process_all_components.py as the main script.
# 
# Functions available:
#   - process_component(component): Process a single velocity component
#   - load, svd, trunc, proj, QR_dec, matrixM, UA_form, VA_form, recon: Helper functions

