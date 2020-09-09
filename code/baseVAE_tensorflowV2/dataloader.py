
import h5py
import numpy as np

class InputHandle():
    def __init__(self, input_param):
    
        self.path_to_x = input_param['path_to_x']
        self.path_to_d = input_param['path_to_d']
        self.path_to_y = input_param['path_to_y']
        
        self.batchsize = input_param['batchsize']
        self.shuffle = input_param['shuffle']
        
        self.x_data, self.d_data, self.y_data, = self.load()
        assert self.x_data.shape == self.y_data.shape
        assert self.x_data.shape[0] == self.d_data.shape[0]
        
        self.N = self.x_data.shape[0]
        self.xdim = self.x_data.shape[1]
        self.ddim = self.d_data.shape[1]
        
        self.init_epoch()

    def load(self):
        x_data = np.load(self.path_to_x)
        d_data = np.load(self.path_to_d)
        y_data = np.load(self.path_to_y)
        return x_data, d_data, y_data

    def samp_batch(self):
        
        if not self.has_batch_left():
            self.init_epoch()
        s = self.curr_idx
        ids = self.indices[s:s+self.batchsize]
        x_batch = self.x_data[ids,:]
        d_batch = self.d_data[ids,:]
        y_batch = self.y_data[ids,:]
        self.curr_idx = s+self.batchsize
        
        return x_batch, d_batch, y_batch

    def init_epoch(self):
        self.curr_idx = 0
        self.indices = np.arange(self.N)
        if self.shuffle:
            np.random.shuffle(self.indices)
                
    def has_batch_left(self):
        start = self.curr_idx
        end = start + self.batchsize
        if end >= self.N:
            return False
        else: 
            return True
            
class InputHandle_():
    def __init__(self, input_param_):
    
        self.path = input_param_['path']
        self.batchsize = input_param_['batchsize']
        self.shuffle = input_param_['shuffle']
        
        self.data = self.load()
        
        self.N, self.xdim = self.data.shape
        
        self.init_epoch()

    def load(self):
        data = np.load(self.path)
        return data

    def samp_batch(self):
        
        if not self.has_batch_left():
            self.init_epoch()
        s = self.curr_idx
        ids = self.indices[s:s+self.batchsize]
        batch = self.data[ids,:]
        self.curr_idx = s+self.batchsize
        
        return batch

    def init_epoch(self):
        self.curr_idx = 0
        self.indices = np.arange(self.N)
        if self.shuffle:
            np.random.shuffle(self.indices)
                
    def has_batch_left(self):
        start = self.curr_idx
        end = start + self.batchsize
        if end >= self.N:
            return False
        else: 
            return True
    