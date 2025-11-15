# George Drakoulas - 170222
# Construct the CAE_2D DL_ model

# hide messages
import warnings

import tensorflow.keras.layers

warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# libraries
from DL_parameters import config

#DL libraries
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Dense,Input,Conv2D,MaxPooling2D,UpSampling2D,Reshape,Flatten
from tensorflow.keras import Model


# 2. DL parameters
latent_space=config['latent_CAE_2D']

# Custom activation function (Swish)
def my_swish(x, beta=1.0):
    return x * K.sigmoid(beta * x)

activ = my_swish
#activ = tensorflow.keras.activations.elu
## Encoder
encoder_inputs = Input(shape=(16, 16,3), name='Field')

x = Conv2D(30, kernel_size=(3, 3), activation=activ, padding='same')(encoder_inputs)
enc_l4 = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

x = Conv2D(20, kernel_size=(3, 3), activation=activ, padding='same')(enc_l4)
enc_l5 = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

x = Conv2D(10, kernel_size=(3, 3), activation=activ, padding='same')(enc_l5)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

x = Flatten()(x)  # Flatten to 1D vector
x = Dense(40, activation=activ)(x)
x = Dense(10, activation=activ)(x)
encoded = Dense(latent_space)(x)
encoder = Model(inputs=encoder_inputs, outputs=encoded)

## Decoder
decoder_inputs = Input(shape=(latent_space,), name='decoded')
x = Dense(10, activation=activ)(decoder_inputs)
x = Dense(40, activation=activ)(x)
x = Dense(2 * 2 * 3, activation=activ)(x)

x = Reshape(target_shape=(2, 2, 3))(x)

x = Conv2D(10, kernel_size=(3, 3), activation=activ, padding='same')(x)  # Convolve
dec_l1 = UpSampling2D(size=(2, 2))(x)  # Upsample

x = Conv2D(20, kernel_size=(3, 3), activation=activ, padding='same')(dec_l1)
dec_l2 = UpSampling2D(size=(2, 2))(x)

x = Conv2D(30, kernel_size=(3, 3), activation=activ, padding='same')(dec_l2)
dec_l3 = UpSampling2D(size=(2, 2))(x)

decoded = Conv2D(3, kernel_size=(3, 3), activation='linear', padding='same')(dec_l3)

decoder = Model(inputs=decoder_inputs, outputs=decoded)

## Autoencoder
ae_outputs = decoder(encoder(encoder_inputs))

model = Model(inputs=encoder_inputs, outputs=ae_outputs, name='CAE')

print(decoder.summary())

