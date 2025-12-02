# George Drakoulas - 170222
# Construct the Algorithm for the Data Preparation of the VAE_FFNN_model.py

# hide messages
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# libraries
import numpy as np
from sklearn.utils import shuffle

class Data_pre():
    def __init__(self, u, time, DATA_1D, sims, stop_time):
        self.u = u
        self.time = time
        self.Data_1D = DATA_1D
        self.sims = sims
        self.stop_time = stop_time

    def data(self):
        # Inputs
        self.inputs = np.zeros((self.Data_1D.shape[0] * self.Data_1D.shape[1], 2))  # 2 = Re,, time

        # Outputs
        self.outputs = np.zeros((self.Data_1D.shape[0] * self.Data_1D.shape[1], self.Data_1D.shape[2]))

        k = 0
        for si in range(self.sims):
            for ti in range(self.stop_time):
                self.inputs[k, 0] = self.time[ti]
                self.inputs[k, 1] = self.u[si]

                self.outputs[k, :] = self.Data_1D[si, ti, :]

                k += 1
        return self.inputs, self.outputs

    def shuffle_data(self):
        self.shuf_x,self.shuf_y=shuffle(self.inputs,self.outputs)

    def split_x(self):
        self.train_x=self.shuf_x[:-10]
        self.test_x=self.shuf_x[-10:-5]
        self.val_x=self.shuf_x[-5:]
        return self.train_x,self.test_x,self.val_x

    def split_y(self):
        self.train_y=self.shuf_y[:-10]
        self.test_y=self.shuf_y[-10:-5]
        self.val_y=self.shuf_y[-5:]
        return self.train_y,self.test_y,self.val_y



