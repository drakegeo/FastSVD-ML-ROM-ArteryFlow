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

# 3. CAE-1D parameters
config['lr_CAE_1D'] = 0.001
config['batch_CAE_1D'] = 4
config['epochs_CAE_1D'] = 1000
config['time_frame_CAE_1D'] = 160  # time until which we train the CONV1D model
config['times_keep_CAE_2D'] = 30 # !! this value determines the output of the FFNN and the time window of the LSTM

# 4. LSTM parameters
config['time_wind_LSTM'] = config['times_keep_CAE_2D'] # time window = number of
config['epochs_LSTM'] = 35000
config['lr_LSTM'] = 0.0001
config['batch_LSTM'] = 5

# 5. FFNN parameters
config['epochs_FFNN'] = 1500
config['lr_FFNN'] = 1
config['batch_FFNN'] = 1


