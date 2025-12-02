## George Drakoulas - 170222
# prepare the Data for the LSTM with inputs and output sequences

import numpy as np

# LSTM data for training
class LSTM_data_tr():
    def __init__(self,file,parameters,time_win,time_frames,latent_space,num_sol):
        self.file=file
        self.time_win=time_win
        self.latent_space=latent_space
        self.num_sol=num_sol
        self.parameters = parameters
        self.time_frames=time_frames

    def matrix(self):
        self.total_size=np.shape(self.file)[0]*np.shape(self.file)[1]
        self.input_seq = np.zeros(shape=(self.total_size - self.time_win * self.num_sol, self.time_win, self.latent_space + 1))
        self.output_seq = np.zeros(shape=(self.total_size - self.time_win * self.num_sol, self.latent_space))
        sample=0

        for snapshot in range(self.num_sol):
            lstm_snapshot = self.file[snapshot][:][:]
            for t in range(self.time_win,  self.time_frames):
                self.input_seq[sample, :, :self.latent_space] = lstm_snapshot[t - self.time_win:t, :]
                self.input_seq[sample, :, self.latent_space:] = self.parameters[snapshot, :]
                self.output_seq[sample, :] = lstm_snapshot[t, :]
                sample += 1
        return self.input_seq,self.output_seq

# LSTM data for testing
class LSTM_data_te():

    def __init__(self,file,parameters,time_win,time_frames,latent_space,num_sol,lstm_model):
        self.file=file
        self.time_win=time_win
        self.latent_space=latent_space
        self.num_sol=num_sol
        self.parameters = parameters
        self.time_frames=time_frames
        self.lstm_model=lstm_model

    def matrix(self):
        self.input_seq = np.zeros(shape=(1, self.time_win, self.latent_space + 1))
        self.output_seq_pred = np.zeros(shape=(self.num_sol,self.time_frames,self.latent_space))

        for snapshot in range(self.num_sol):
            self.input_seq[0,:,:-1] = self.file[snapshot,0:self.time_win,:]
            self.input_seq[0,:,-1] = self.parameters[snapshot,0]

            self.output_seq_pred[snapshot,:self.time_win,:]=self.file[snapshot,:self.time_win,:]

            for t in range(self.time_win,self.time_frames):
                self.output_seq_pred[snapshot, t, :] = self.lstm_model.predict(self.input_seq[0:1, :, :])[0, :]
                self.input_seq[0, 0:self.time_win - 1, :-1] = self.input_seq[0, 1:, :-1]
                self.input_seq[0, self.time_win - 1, :-1] = self.output_seq_pred[snapshot, t, :]
        return self.output_seq_pred

