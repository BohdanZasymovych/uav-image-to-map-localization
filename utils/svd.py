import numpy as np
from numpy.typing import NDArray

def _compute_jacobi_rotation(col_i: NDArray, col_j: NDArray, tol: float = 1e-12) -> tuple[float, float, bool]:
    """
    Computes the cosine and sine values for a Jacobi rotation 
    that orthogonalizes two columns.
    """
    alpha = np.dot(col_i, col_i)
    beta = np.dot(col_j, col_j)
    gamma = np.dot(col_i, col_j)

    if abs(gamma) < tol:
        return 1.0, 0.0, False

    zeta = (beta - alpha) / (2.0 * gamma)
    
    if zeta == 0.0:
        t = 1.0
    else:
        t = np.sign(zeta) / (abs(zeta) + np.sqrt(1.0 + zeta**2))
        
    c = 1.0 / np.sqrt(1.0 + t**2)
    s = c * t
    
    return c, s, True

def _apply_givens_rotation(M: NDArray, i: int, j: int, c: float, s: float) -> None:
    """
    Applies a Givens rotation to columns i and j of matrix M in-place.
    """
    col_i = M[:, i].copy()
    col_j = M[:, j].copy()
    
    M[:, i] = c * col_i - s * col_j
    M[:, j] = s * col_i + c * col_j

def compute_svd_jacobi(A: NDArray, max_sweeps: int = 100, tol: float = 1e-12) -> tuple[NDArray, NDArray, NDArray]:
    """
    Computes the Singular Value Decomposition (SVD) of a matrix A = U * S * V^T 
    using the One-Sided Jacobi algorithm. 
    
    This method iteratively orthogonalizes the columns of A using Givens rotations 
    until convergence, without explicitly computing A^T @ A, thereby preserving 
    numerical stability.
    
    Parameters
    ----------
    A : NDArray
        Input matrix of shape (M, N).
    max_sweeps : int
        Maximum number of iterative sweeps over all column pairs.
    tol : float
        Tolerance for determining column orthogonality.
        
    Returns
    -------
    U : NDArray
        Left singular vectors matrix of shape (M, N).
    S : NDArray
        Singular values array of shape (N,) sorted in descending order.
    Vt : NDArray
        Transposed right singular vectors matrix of shape (N, N).
    """
    M_rows, N_cols = A.shape
    
    W = A.astype(np.float64).copy()
    V = np.eye(N_cols, dtype=np.float64)
    
    for _ in range(max_sweeps):
        changed = False
        for i in range(N_cols - 1):
            for j in range(i + 1, N_cols):
                c, s, rotated = _compute_jacobi_rotation(W[:, i], W[:, j], tol)
                if rotated:
                    _apply_givens_rotation(W, i, j, c, s)
                    _apply_givens_rotation(V, i, j, c, s)
                    changed = True
                    
        if not changed:
            break

    S = np.linalg.norm(W, axis=0)
    
    U = np.zeros_like(W)
    for i in range(N_cols):
        if S[i] > tol:
            U[:, i] = W[:, i] / S[i]
            
    sorted_indices = np.argsort(S)[::-1]
    S = S[sorted_indices]
    U = U[:, sorted_indices]
    V = V[:, sorted_indices]
    
    return U, S, V.T
