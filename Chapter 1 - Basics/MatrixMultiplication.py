import numpy as np

def matmul_forward(X: np.ndarray,
                   W: np.ndarray) -> np.ndarray:
    """
    Function computes forward pass of a matrix multiplication
    """
    
    assert X.shape[1] == W.shape[0],\
        """
        Count of columns in first matrix must be the same as
        count of rows in second matrix
        """
    
    N = np.dot(X, W)
    return N

def matmul_backward_first(X: np.ndarray,
                          W: np.ndarray) -> np.ndarray:
    """
    Function computes backward pass of matrix multiplication
    with respect to the first argument.
    """
    
    dN_dX = np.transpose(W, (1, 0))
    return dN_dX

if __name__ == "__main__":
    
    np.random.seed(113344)
    
    X = np.random.randn(1, 5)
    W = np.random.randn(5, 1)
    
    print(X)
    print(matmul_backward_first(X, W))