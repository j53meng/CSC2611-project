import numpy as np
from sklearn.neighbors import NearestNeighbors

def calculate_id_pr(word_vector, neighborhood_vectors):
    """
    Calculates the Participation Ratio (ID_PR) for a word's neighborhood.
    Based on ID_PR = 1 / sum((lambda_k / V)^2)
    """
    # 1. Center the neighborhood matrix (M)
    M = neighborhood_vectors - np.mean(neighborhood_vectors, axis=0)
    
    # 2. Singular Value Decomposition
    _, sigmas, _ = np.linalg.svd(M, full_matrices=False)
    
    # 3. Variance (lambda) = sigma^2
    lambdas = sigmas**2
    V = np.sum(lambdas) # Total Variance
    
    # 4. Participation Ratio Formula
    denominator = np.sum((lambdas / V)**2)
    id_pr = 1 / denominator
    
    return id_pr