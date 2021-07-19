# Evaluation metrics: 
# > RMSE&ΔRMSE;
# > Correlation;
# > IDR
# > Positive Precision@k and negative Precision@k
# > rank in HTVS & ΔRank; 

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import IDR
import random
random.seed(123)
    
def calu_rmse(real_arr, pred_arr, mode='feature'):
    """RMSE and R-median-se.
    
    :param real_arr: numpy array sample x genes
    :param pred_arr: numpy array sample x genes
    :param mode: to be 'sample' or 'feature'
    :return: mse array, rmse
    """
    if mode=='feature':
        feature_mse_arr = mean_squared_error(real_arr, pred_arr, multioutput='raw_values')
        mse = np.mean(feature_mse_arr)
        return feature_mse_arr, np.sqrt(mse)
    elif mode=='sample':
        samp_mse_arr = mean_squared_error(real_arr.T, pred_arr.T, multioutput='raw_values')
        mse = np.mean(samp_mse_arr)
        return samp_mse_arr, np.sqrt(mse)
    else:
        return None
    
def calu_deltaMSE(real_arr, pred_arr_before_train, pred_arr):
    """feature-wise delta mse
    :param real_arr: numpy array sample x genes
    :param pred_arr: numpy array sample x genes
    :param pred_arr_before_train: numpy array sample x genes
    :return: array: len==feature
    """
    pred_mse_arr = mean_squared_error(real_arr, pred_arr, multioutput='raw_values')
    before_train_mse_arr = mean_squared_error(real_arr, pred_arr_before_train, multioutput='raw_values')
    delta_mse_arr = pred_mse_arr - before_train_mse_arr
    return delta_mse_arr
    
def calu_corr(real_arr, pred_arr, correlation_type='pearson'):
    """Correlation between real and predicted.
    :param real_arr: numpy array sample x genes
    :param pred_arr: numpy array sample x genes
    :correlation_type: 'pearson' or 'spearman'
    :return: score array, mean_score. 
    
    """
    if correlation_type == 'pearson':
        corr = pearsonr
    elif correlation_type == 'spearman':
        corr = spearmanr
    else:
        raise ValueError("Unknown correlation type: %s" % correlation_type)
    assert len(real_arr)==len(pred_arr)
    score = []
    for r in range(len(real_arr)):
        score.append(corr(real_arr[r], pred_arr[r])[0])
    return score, np.mean(score)
    
def calu_precison_topk(real_arr, pred_arr, num_pos=50, num_neg=50):
    """Precison@k.
    :param real_arr: numpy array sample x genes
    :param pred_arr: numpy array sample x genes
    :return: intersection percentage of top k pos and top k neg.
    """
    precision_k_neg = []
    precision_k_pos = []
    real_arr = np.argsort(real_arr, axis=1) #argsort之后, top k所在的index为: real_arr[0:k]
    pred_arr = np.argsort(pred_arr, axis=1)
    # pos
    pos_real_set = real_arr[:, -num_pos:]
    pos_pred_set = pred_arr[:, -num_pos:]
    assert len(pos_real_set)==len(pos_pred_set)
    for i in range(len(pos_real_set)):
        pos_real = set(pos_real_set[i])
        pos_pred = set(pos_pred_set[i])
        precision_k_pos.append(len(pos_real.intersection(pos_pred)) / num_pos)
    # neg    
    neg_real_set = real_arr[:, :num_neg]
    neg_pred_set = pred_arr[:, :num_neg]
    assert len(neg_real_set)==len(neg_pred_set)
    for i in range(len(neg_real_set)):
        neg_real = set(neg_real_set[i])
        neg_pred = set(neg_pred_set[i])
        precision_k_neg.append(len(neg_real.intersection(neg_pred)) / num_neg)
    return precision_k_pos, precision_k_neg
    
def calu_idr(real_arr, pred_arr):
    """Irreproducible Discovery Rate. 'rank' consistancy between two replicates. we will use the R package 'idr', p.s.: results from python package pyidr is not reliable.
    :param real_arr: numpy array sample x genes
    :param pred_arr: numpy array sample x genes
    :return: list, result for each sample.
    """
    idr=IDR.IDR()
    result=[]
    for i in range(len(real_arr)):
        arr_1 = real_arr[i]
        arr_2 = pred_arr[i]
        res_p, res_rho = idr.fit(arr_1, arr_2)
        result.append(res_p[0])
        #arr_1_indicies = np.argsort(arr_1)
        #arr_1_rank = np.array([arr_1_indicies.tolist().index(i) for i in range(len(a))])
        #b_indicies = np.argsort(b)
        #b_rank = np.array([b_indicies.tolist().index(i) for i in range(len(b))])
    return result
    
#def calu_rank():
#    return
