## George Drakoulas - 170222
# Parameters for the Deep Learning
config = dict()

# 1. Parameters for SVD
# Truncation errors for each component
config['trunc_error_x'] = 8.4e-6 # 9-> 251 DONE!
config['trunc_error_y'] =  9.8e-6 # 8.5 -> 260
config['trunc_error_z'] =  6.2e-6 # less keep more

config['time_sol'] = 300  # time until which we keep the solution,derived through the post-processing -> post processing

# 2. CAE-2D parameters
config['lr_CAE_2D'] = 0.0005#0.0001
config['batch_CAE_2D'] = 20
config['epochs_CAE_2D'] = 2000
config['latent_CAE_2D'] = 4 # 4 and 5 almost same

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


