import argparse
import pandas as pd
import numpy as np
from metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('-real', type=str, default=None)
parser.add_argument('-pred', type=str, default=None)
parser.add_argument('-out_path', type=str, default=None)
parser.add_argument('-out_name', type=str, default=None)
args = parser.parse_args()

def dataload(filename):
    """
    args:
        filename: input file.
    Returns:
        np.narray"""
    if filename.split(".")[-1]=='csv':
        data=pd.read_csv(filename)
        data=data.values
    elif filename.split(".")[-1]=='npy':
        data=np.load(filename)
    else:
        print("invalid input file type")
    return data.astype(np.float32)
        
real_arr=dataload(args.real)
pred_arr=dataload(args.pred)

## calu
rmse_arr,_ = calu_rmse(real_arr, pred_arr, mode='sample')
corr_arr,_ = calu_corr(real_arr, pred_arr, correlation_type='pearson')
precision_k_pos, precision_k_neg = calu_precison_topk(real_arr, pred_arr, num_pos=50, num_neg=50)
idr = calu_idr(real_arr, pred_arr)

## output
df_out=pd.DataFrame({'MSE':rmse_arr,
                    'Pearson':corr_arr,
                    'precision@50_pos':precision_k_pos,
                    'precision@50_neg':precision_k_neg,
                    'IDR':idr})
df_out.to_csv(os.path.join(args.out_path, args.out_name))