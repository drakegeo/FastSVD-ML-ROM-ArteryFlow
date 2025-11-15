"""
Utility script to extract x, y, z velocity components from .dat files.

The .dat files contain all three velocity components. This script extracts
and saves them separately as .npy files for use in the SVD update process.
"""
import numpy as np
import os
from pathlib import Path
import natsort

def extract_components_from_dat(dat_file, output_dir, data_format='concatenated'):
    """
    Extract x, y, z velocity components from a .dat file.
    
    Parameters:
    -----------
    dat_file : str or Path
        Path to the .dat file containing velocity data
    output_dir : str or Path
        Directory to save the extracted .npy files (not used, kept for compatibility)
    data_format : str
        Format of data in .dat file:
        - 'concatenated': [all_x, all_y, all_z] (default)
        - 'interleaved': [x1, y1, z1, x2, y2, z2, ...]
    
    Returns:
    --------
    dict : Dictionary with keys 'x', 'y', 'z' containing numpy arrays
    """
    # Load data
    data = np.loadtxt(dat_file)
    total_points = len(data) // 3
    
    if data_format == 'concatenated':
        # Format: [all_x_values, all_y_values, all_z_values]
        u_x = data[:total_points]
        u_y = data[total_points:2*total_points]
        u_z = data[2*total_points:3*total_points]
    elif data_format == 'interleaved':
        # Format: [x1, y1, z1, x2, y2, z2, ...]
        u_x = data[0::3]  # Every 3rd element starting at 0
        u_y = data[1::3]  # Every 3rd element starting at 1
        u_z = data[2::3]  # Every 3rd element starting at 2
    else:
        raise ValueError(f"Unknown data_format: {data_format}. Use 'concatenated' or 'interleaved'")
    
    return {'x': u_x, 'y': u_y, 'z': u_z}


