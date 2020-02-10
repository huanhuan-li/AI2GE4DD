
import h5py
import numpy as np

def __init__(self, input_param):
    
    self.path = input_param['path']
    self.is_training = input_param['is_training']
    self.batchsize = input_param['batchsize']
    self.val_N = input_param['val_N']
    
    self.all_data = self.load()
    self.train_data, self.val_data = self.split_dataset()
    
    if self.is_training:
        self.shuffle = True
        self.data = self.train_data
        
    else: #for validation
        self.shuffle = False
        self.data = self.val_data
        
    self.N, self.xdim = self.data.shape

def load(self):
    if self.path.split(".")[-1] == 'hdf5':
        h5data = h5py.File(self.path, 'r')
        data = h5data['expr']['block0_values'][:]
        samp_ids = h5data['expr']['block0_values'][:]
    return data, samp_ids
    
def split_dataset(self):
    idxs = np.arange(self.all_data.shape[0])
    np.random.shuffle(idxs)
    val_data = self.data[idxs[0:self.val_N], :]
    train_data = self.data[idxs[self.val_N:self.all_data.shape[0]], :]
    return train_data, val_data

def samp_batch(self):
    if not self.has_batch_left():
        self.init_batch()
    s = self.curr_idx
    ids = self.indices[s:s+self.batchsize]
    x_batch = self.data[ids,:]
    self.curr_idx = s+self.batchsize
    return x_batch

def init_batch(self):
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