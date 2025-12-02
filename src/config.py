## George Drakoulas - 170222
# Parameters for the Deep Learning
config = dict()

# 1. Parameters for SVD
# Truncation errors for each component
# Note: All bases will be standardized to 256 modes via standardize_basis_sizes.py
# These truncation errors control initial POD calculation, then bases are adjusted to 256
config['trunc_error_x'] = 5.0e-4
config['trunc_error_y'] = 5.75e-4
config['trunc_error_z'] = 4.0e-4

# Target number of modes for all components (used by standardize_basis_sizes.py)
config['target_basis_modes'] = 256
config['time_sol'] = 160  # time until which we keep the solution,derived through the post-processing -> post processing

# 2. CAE-2D parameters
config['lr_CAE_2D'] = 0.0005  # Learning rate
config['batch_CAE_2D'] = 20  # Batch size
config['epochs_CAE_2D'] = 2000  # Number of training epochs
config['latent_CAE_2D'] = 4  # Latent space dimension (4 and 5 almost same)
config['val_split_CAE_2D'] = 0.1  # Validation split ratio (10% of training data)
config['spatial_shape_CAE_2D'] = (16, 16)  # Spatial dimensions for reshaping

# 3. LSTM parameters
config['epochs_LSTM'] = 3000  # Number of training epochs
config['lr_LSTM'] = 0.0001  # Learning rate
config['batch_LSTM'] = 5  # Batch size
config['val_split_LSTM'] = 0.1  # Validation split ratio (10% of training data)
config['time_window_LSTM'] = config['time_sol']  # Time window = time_sol (160)

# 4. FFNN parameters
config['epochs_FFNN'] = 1500  # Number of training epochs
config['lr_FFNN'] = 1  # Learning rate
config['batch_FFNN'] = 1  # Batch size
config['FFNN_time_window'] = 30  # Time window = number of first timesteps to use
config['val_split_FFNN'] = 0.1  # Validation split ratio (10% of training data)


