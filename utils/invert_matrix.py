import numpy as np
from numpy.typing import NDArray

def invert_matrix(A: NDArray) -> NDArray:
    """
    Inverts a square matrix using Gaussian elimination (augmenting with Identity).
    
    Parameters
    ----------
    A : NDArray, shape (N, N)
    
    Returns
    -------
    A_inv : NDArray, shape (N, N)
    """
    n = A.shape[0]
    AI = np.hstack([A.astype(np.float64), np.eye(n, dtype=np.float64)])
    
    for i in range(n):
        pivot_idx = i + np.argmax(np.abs(AI[i:, i]))
        if AI[pivot_idx, i] == 0:
            raise np.linalg.LinAlgError("Singular matrix: cannot invert.")
        
        AI[[i, pivot_idx]] = AI[[pivot_idx, i]]
        
        AI[i] = AI[i] / AI[i, i]
        
        for j in range(n):
            if i != j:
                factor = AI[j, i]
                AI[j] -= factor * AI[i]
                
    return AI[:, n:]
