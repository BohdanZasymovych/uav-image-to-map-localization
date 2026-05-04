import numpy as np
import math

def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    """
    Computes the Cholesky decomposition of a matrix.
    Acts as a drop-in manual replacement for scipy.linalg.cho_factor.
    
    Parameters:
    a (array_like): Square, symmetric positive-definite matrix.
    lower (bool, optional): Whether to compute the lower or upper factor. Default is False (upper).
    overwrite_a (bool, optional): Whether to overwrite data in a (may improve performance).
    check_finite (bool, optional): Whether to check that the input contains only finite numbers.
    
    Returns:
    c (ndarray): Matrix whose upper or lower triangle contains the Cholesky factor.
    lower (bool): Flag indicating whether the factor is in the lower or upper triangle.
    """
    A = np.array(a, copy=not overwrite_a, dtype=float)
    
    if check_finite and not np.isfinite(A).all():
        raise ValueError("Array must not contain infs or NaNs.")
    
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Expected a square matrix.")


    L = np.zeros((n, n), dtype=float)
    
    for i in range(n):
        for j in range(i + 1):
            
            sum_k = sum(L[i][k] * L[j][k] for k in range(j))
            
            if i == j:
                val = A[i][i] - sum_k

                if val <= 0:
                    raise np.linalg.LinAlgError("Matrix is not positive definite")
                
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (1.0 / L[j][j]) * (A[i][j] - sum_k)

    if lower:
        return L, True
    else:
        U = L.T
        return U, False
