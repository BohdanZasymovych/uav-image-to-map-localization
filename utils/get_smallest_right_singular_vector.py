import numpy as np
from utils.cho_solve import cho_solve
from utils.cho_factor import cho_factor


def get_smallest_right_singular_vector(A, tol=1e-6, maxiter=1000, shift=1e-8):
    """
    Finds the right singular vector corresponding to the smallest singular value of A.
    
    Parameters:
    A (ndarray): The input matrix (m x n).
    tol (float): Tolerance for convergence.
    maxiter (int): Maximum number of iterations.
    shift (float): Small regularization term to ensure positive-definiteness.
    
    Returns:
    v (ndarray): The target right singular vector of length n.
    """
    n_cols = A.shape[1]
    
    M = (A.T @ A) + shift * np.eye(n_cols)
    
    c, lower_flag = cho_factor(M, lower=True)
    
    v = np.random.randn(n_cols)
    v /= np.linalg.norm(v)
    
    for _ in range(maxiter):

        v_new = cho_solve((c, lower_flag), v)
        
        v_new /= np.linalg.norm(v_new)
        
        if np.linalg.norm(v_new - v) < tol or np.linalg.norm(v_new + v) < tol:
            return v_new
            
        v = v_new
        
    return v
