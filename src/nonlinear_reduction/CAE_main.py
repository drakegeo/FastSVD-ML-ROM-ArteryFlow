# George Drakoulas - 170222
# Construct the CAE_2D DL_ model

# hide messages
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# George Drakoulas - 170222
# Construct the CAE_2D main algorithm

# libraries
import matplotlib.pyplot as plt
from libraries_plot import *
from useful_func import Reshape_, Standarization,Normalization
from DL_parameters import config
import numpy as np
from CAE_model import *
#DL libraries
from tensorflow.keras import  optimizers
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import pandas as pd

filepath = './train_ux_POD'
name_sim = []
for file in os.listdir(filepath):
    print(file)

# 1. CAE_2D_params
lr_rate=config['lr_CAE_2D']
batch_size=config['batch_CAE_2D']
epochs=config['epochs_CAE_2D']

########## x-axis
data_x=np.load('./fnl_snapshot/vel_snapshot_x.npy')
print(np.shape(data_x))

print(data_x.shape)
fnl_mat_x = data_x[0][:][:]
for i in range(1, np.shape(data_x)[0]):
    fnl_mat_x = np.append(fnl_mat_x, data_x[i][:][:], axis=1)
print(fnl_mat_x.shape)

# 3. Data Standardization
stand_data_x= Standarization(fnl_mat_x)
stdmean=stand_data_x.param()
np.save('./scaling_data/stdmean_x_CAE2D.npy',stdmean)
scale_snapshot_x=stand_data_x.stand().T
print(scale_snapshot_x.shape)
quit()
# reshape the scaled snapshot
res_data_x = []
for i in range(1600):
    re_x = scale_snapshot_x[i,:].reshape(16,16)
    res_data_x.append(re_x)


########## y-axis
data_y=np.load('./fnl_snapshot/vel_snapshot_y.npy')

print(data_y.shape)
fnl_mat_y = data_y[0][:][:]
for i in range(1, np.shape(data_y)[0]):
    fnl_mat_y = np.append(fnl_mat_y, data_y[i][:][:], axis=1)
print(fnl_mat_y.shape)

# 3. Data Standardization
stand_data_y= Standarization(fnl_mat_y)
stdmean=stand_data_y.param()
np.save('./scaling_data/stdmean_y_CAE2D.npy',stdmean)
scale_snapshot_y=stand_data_y.stand().T
print(scale_snapshot_y.shape)

# reshape the scaled snapshot
res_data_y = []
for i in range(1600):
    re_y = scale_snapshot_y[i,:].reshape(16,16)
    res_data_y.append(re_y)

########## z-axis
data_z=np.load('./fnl_snapshot/vel_snapshot_z.npy') #

print(data_z.shape)
fnl_mat_z = data_z[0][:][:]
for i in range(1, np.shape(data_z)[0]):
    fnl_mat_z = np.append(fnl_mat_z, data_z[i][:][:], axis=1)
print(fnl_mat_z.shape)

# 3. Data Standardization
stand_data_z= Standarization(fnl_mat_z) #USE NORMALIZATION
stdmean=stand_data_z.param()
np.save('./scaling_data/stdmean_z_CAE2D.npy',stdmean)
scale_snapshot_z=stand_data_z.stand().T
print(scale_snapshot_z.shape)


# reshape the scaled snapshot
res_data_z = []
for i in range(1600):
    re_z = scale_snapshot_z[i,:].reshape(16,16)
    res_data_z.append(re_z)

print(np.shape(res_data_z))

### final matrix construction
final_mat = np.stack([res_data_x,res_data_y,res_data_z],3)
print(final_mat.shape)

fig, ax = plt.subplots(3)
ax[0].plot(final_mat[:160,2,2,0])
ax[1].plot(final_mat[:160,2,2,1])
ax[2].plot(final_mat[:160,2,2,2])

ax[0].plot(final_mat[160:2*160,2,2,0])
ax[1].plot(final_mat[160:2*160,2,2,1])
ax[2].plot(final_mat[160:2*160,2,2,2])

ax[0].plot(final_mat[2*160:3*160,2,2,0])
ax[1].plot(final_mat[2*160:3*160,2,2,1])
ax[2].plot(final_mat[2*160:3*160,2,2,2])

ax[0].plot(final_mat[3*160:4*160,2,2,0])
ax[1].plot(final_mat[3*160:4*160,2,2,1])
ax[2].plot(final_mat[3*160:4*160,2,2,2])

ax[0].plot(final_mat[4*160:5*160,2,2,0])
ax[1].plot(final_mat[4*160:5*160,2,2,1])
ax[2].plot(final_mat[4*160:5*160,2,2,2])

ax[0].plot(final_mat[6*160:7*160,2,2,0])
ax[1].plot(final_mat[6*160:7*160,2,2,1])
ax[2].plot(final_mat[6*160:7*160,2,2,2])

ax[0].plot(final_mat[7*160:8*160,2,2,0])
ax[1].plot(final_mat[7*160:8*160,2,2,1])
ax[2].plot(final_mat[7*160:8*160,2,2,2])


ax[0].plot(final_mat[8*160:9*160,2,2,0])
ax[1].plot(final_mat[8*160:9*160,2,2,1])
ax[2].plot(final_mat[8*160:9*160,2,2,2])

plt.show()

# 5.  Design the CAE_2D network
weights_filepath='./DL_weights/weights_CAE2D.h5'
my_adam = optimizers.Adam(lr=lr_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #optimizer
# Save the weights only if the validation loss reduces from previous best
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',
                            save_weights_only=True)

# Use an early-stopping criterion for preventing overfitting on training data
earlystopping = EarlyStopping(monitor='val_loss',  patience=50, verbose=1)
callbacks_list = [checkpoint, earlystopping]

# 5. Fit the network
model.compile(optimizer=my_adam, loss='mean_squared_error')  # Use a simple L-2 norm for training the autoencoder
model.summary()

# 6. Train the network
with tf.device('/gpu:0'):
    train_history = model.fit(x=final_mat,
                              y=final_mat,
                              epochs=25000, batch_size=32, # increase epochs to 50000
                               validation_split=0.1,callbacks=None)
    encoder.save_weights('./DL_weights/enc_CAE2D.h5')
    decoder.save_weights('./DL_weights/dec_CAE2D.h5')

# 7. save the DL_trained data to DL_dat folder
decoded_data=decoder.predict(encoder.predict(final_mat))
encoded_data=encoder(final_mat)

np.save('./DL_data/CAE2D_dec.npy',decoded_data)
np.save('./DL_data/CAE2D_enc.npy',encoded_data)

# 8. save the results in a csv file
df_results=pd.DataFrame(train_history.history)
df_results['epoch']=train_history.epoch
df_results.to_csv(path_or_buf='./results_csv/CAE_2D.csv',index=False)

# "Loss"
plt.plot(train_history.history['loss'], )
plt.plot(train_history.history['val_loss'])
plt.title('$model loss$')
plt.ylabel('$loss$')
plt.xlabel('$epoch$')
plt.legend(['$train$', '$validation$'], loc='upper left')
plt.show()


