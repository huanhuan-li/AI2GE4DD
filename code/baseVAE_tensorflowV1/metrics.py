# coding=utf-8

import numpy as np
#import tensorflow as tf
#from sklearn.metrics import mutual_info_score
#from sklearn.ensemble import RandomForestRegressor

'''generative factor is unknown'''
### D(q(z)||p(z))
### D(q(z)||q^(z)), q^(z) = ∏_j[q(z_j)]
### Corr(z_i, z_j)
# MMD distance: https://github.com/ShengjiaZhao/InfoVAE/blob/master/mmd_vae.py
# sample from q^(z): sample from q(z), then shuffle dim by dim (FactorVAE)
def compute_kernel(x, y):
    x_size, dim = x.shape
    y_size = y.shape[0]
    tiled_x = np.tile(x.reshape([x_size, 1, dim]), [1, y_size, 1])
    tiled_y = np.tile(y.reshape([1, y_size, dim]), [x_size, 1, 1])
    return np.exp(-np.mean(np.square(tiled_x - tiled_y), axis=2) / dim)
    
def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return np.mean(x_kernel) + np.mean(y_kernel) - 2 * np.mean(xy_kernel)

# KL(q(z_i|x)||p(z_i)). indicate capacity of latent channel z_i, which is i-th dimension of z
def comput_capacity(z_mean, z_logvar):
    assert z_mean.shape == z_logvar.shape
    return np.mean(0.5 * (np.square(z_mean) + np.exp(z_logvar) - z_logvar - 1), 0)

### MI(z_i, encoder(z))
# to be added

'''generative factor is known'''  
'''      
# 1. Mutual Information Gap(MIG)
def compute_MIG(ground_truth_data,
                representation_function,
                random_state,
                num_train=gin.REQUIRED,
                batch_size):
    """
    Computes the mutual information gap.
    Args:
        ground_truth_data: GroundTruthData to be sampled from.
        representation_function: Function that takes observations as input and outputs a dim_representation sized representation for each observation.
        random_state: Numpy random state used for randomness.
        artifact_dir: Optional path to directory where artifacts can be saved.
        num_train: Number of points used for training.
        batch_size: Batch size for sampling.
    Returns:
        Dict with average mutual information gap.
    """
    mus_train, ys_train = utils.generate_batch_factor_code(ground_truth_data, representation_function, num_train, random_state, batch_size)
    assert mus_train.shape[1] == num_train
    return _compute_MIG(mus_train, ys_train)

# estimate the discrete mutual information by binning each dimension of the representations obtained from 10,000 points into 20 bins;
def _compute_MIG(z, v):
    """Computes score based on both training and testing codes and factors. v->x, x->z"""
    score_dict = {}
    discretized_z = utils.make_discretizer(z)
    m = discrete_mutual_info(discretized_z, v)
    assert m.shape[0] == z.shape[0]
    assert m.shape[1] == v.shape[0]
    # m is [num_latents, num_factors]
    entropy = discrete_entropy(v)
    sorted_m = np.sort(m, axis=0)[::-1]
    score_dict["discrete_MIG"] = np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
    return score_dict
  
def discrete_mutual_info(z, v):
    """Compute discrete mutual information. v->x, x->z"""
    num_codes = z.shape[0]
    num_factors = v.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = sklearn.metrics.mutual_info_score(v[j, :], z[i, :])
    return m
  
def discrete_entropy(v):
    """Compute discrete mutual information."""
    num_factors = v.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = sklearn.metrics.mutual_info_score(v[j, :], v[j, :])
    return h
    
def generate_batch_factor_code(ground_truth_data, representation_function,
                               num_points, random_state, batch_size):
    """
    Sample a single training sample based on a mini-batch of ground-truth data.
    Args:
        ground_truth_data: GroundTruthData to be sampled from.
        representation_function: Function that takes observation as input and outputs a representation.
        num_points: Number of points to sample.
        random_state: Numpy random state used for randomness.
        batch_size: Batchsize to sample points.
    Returns:
        representations: Codes (num_codes, num_points)-np array. 
        factors: Factors generating the codes (num_factors, num_points)-np array.
    """
    representations = None
    factors = None
    i = 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_factors, current_observations = ground_truth_data.sample(num_points_iter, random_state)
        if i == 0:
            factors = current_factors
            representations = representation_function(current_observations)
        else:
            factors = np.vstack((factors, current_factors))
            representations = np.vstack((representations, 
                                   representation_function(
                                       current_observations)))
        i += num_points_iter
    return np.transpose(representations), np.transpose(factors)
    
# DCI disentanglement

def compute_dci(ground_truth_data, 
                representation_function, 
                random_state,
                num_train,
                num_test,
                batch_size=16):
    """
    Computes the DCI scores.
    Args:
        ground_truth_data: GroundTruthData to be sampled from.
        representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
        random_state: Numpy random state used for randomness.
        num_train: Number of points used for training.
        num_test: Number of points used for testing.
        batch_size: Batch size for sampling.
    Returns:
        Dictionary with average disentanglement score, completeness and informativeness (train and test).
    """
    # c_train are of shape [num_codes, num_train], while f_train are of shape [num_factors, num_train].
    c_train, f_train = utils.generate_batch_factor_code(
      ground_truth_data, representation_function, num_train,
      random_state, batch_size)
  assert mus_train.shape[1] == num_train
  assert ys_train.shape[1] == num_train
  mus_test, ys_test = utils.generate_batch_factor_code(
      ground_truth_data, representation_function, num_test,
      random_state, batch_size)
  scores = _compute_dci(mus_train, ys_train, mus_test, ys_test)
  return scores

def _compute_dci(mus_train, ys_train, mus_test, ys_test):
    """Computes score based on both training and testing codes and factors."""
    scores = {}
    importance_matrix, train_err, test_err = compute_importance_gbt(mus_train, ys_train, mus_test, ys_test)
    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    scores["informativeness_train"] = train_err
    scores["informativeness_test"] = test_err
    scores["disentanglement"] = disentanglement(importance_matrix)
    scores["completeness"] = completeness(importance_matrix)
    return scores

def compute_importance_gbt(c_train, f_train, c_test, f_test):
    """Compute importance based on gradient boosted trees."""
    num_factors = f_train.shape[0]
    num_codes = c_train.shape[0]
    importance_matrix = np.zeros(shape=[num_codes, num_factors],
                                dtype=np.float64)
    train_loss = []
    test_loss = []
    for i in range(num_factors):
        model = GradientBoostingRegressor()
        model.fit(c_train.T, f_train[i, :])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(c_train.T) == f_train[i, :]))
        test_loss.append(np.mean(model.predict(c_test.T) == f_test[i, :]))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)

def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                    base=importance_matrix.shape[1])
                                    
def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()
    return np.sum(per_code*code_importance)
    
def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                    base=importance_matrix.shape[0])

def completeness(importance_matrix):
    """"Compute completeness of the representation."""
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor*factor_importance)
'''