# BetaVAE score
# FactorVAE score
# Mutual Information Gap(MIG)
# DCI disentanglement

from sklearn.metrics import mutual_info_score
        
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
  