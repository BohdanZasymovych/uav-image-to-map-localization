import numpy as np
from numpy.typing import NDArray

def solve_linear_system(A: NDArray, b: NDArray) -> NDArray:
    """
    Solves a linear system Ax = b using Gaussian elimination with partial pivoting.
    
    Parameters
    ----------
    A : NDArray, shape (N, N)
    b : NDArray, shape (N,)
    
    Returns
    -------
    x : NDArray, shape (N,)
    """
    n = A.shape[0]
    Ab = np.hstack([A.astype(np.float64), b.astype(np.float64).reshape(-1, 1)])
    
    for i in range(n):
        pivot_idx = i + np.argmax(np.abs(Ab[i:, i]))
        if Ab[pivot_idx, i] == 0:
            raise np.linalg.LinAlgError("Singular matrix: linear system cannot be solved.")
            
        Ab[[i, pivot_idx]] = Ab[[pivot_idx, i]]
        
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]
            
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, n] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
        
    return x
