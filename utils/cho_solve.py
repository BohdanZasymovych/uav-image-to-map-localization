import numpy as np


def forward_substitution(L, b):
    """Solves L * y = b for a lower triangular matrix L."""
    n = len(b)
    y = np.zeros(n, dtype=float)
    
    for i in range(n):
        sum_Ly = b[i] 
        
        for j in range(i):
            sum_Ly -= L[i][j] * y[j]
            
        y[i] = sum_Ly / L[i][i]
        
    return y


def backward_substitution(U, y):
    """Solves U * x = y for an upper triangular matrix U."""
    n = len(y)
    x = np.zeros(n, dtype=float)
    
    for i in range(n - 1, -1, -1):
        sum_Ux = y[i]
        
        for j in range(i + 1, n):
            sum_Ux -= U[i][j] * x[j]
            
        x[i] = sum_Ux / U[i][i]
        
    return x


def cho_solve(c_and_lower, b):
    """
    Solves the linear equations A * x = b, given the Cholesky factorization of A.
    Acts as a drop-in manual replacement for scipy.linalg.cho_solve.
    
    Parameters:
    c_and_lower (tuple): (matrix, bool) returned by cho_factor.
    b (array_like): Right-hand side vector.
    
    Returns:
    x (ndarray): The solution to the system A * x = b.
    """
    c, lower = c_and_lower
    
    if lower:
        L = c
        U = c.T
    else:
        U = c
        L = c.T
        
    y = forward_substitution(L, b)
    
    x = backward_substitution(U, y)
    
    return x