import numpy as np
from typing import Callable
import MathFunctions as myMathFunctions

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

Array_Function = Callable[[np.ndarray], np.ndarray]

def matrix_forward_extra(X: np.ndarray,
                         W: np.ndarray,
                         sigma: Array_Function) -> np.ndarray:
    
    """
    Function computes forward pass of function involving matrix 
    multiplication, one extra function
    """
    
    assert X.shape[1] == W.shape[0],\
        """
        Count of columns in first matrix must be the same as
        count of rows in second matrix
        """
        
    N = np.dot(X, W)
    S = sigma(N)
    
    return S

def matrix_function_backward_1(X: np.ndarray,
                                   W: np.ndarray,
                                   sigma: Array_Function) -> np.ndarray:
    """
    Function calculates derivative of our matrix function with respect 
    to the first element.
    """
    
    assert X.shape[1] == W.shape[0],\
        """
        Count of columns in first matrix must be the same as
        count of rows in second matrix
        """
        
    N = np.dot(X,W)
    S = sigma(N)
    
    dS_dN = myMathFunctions.deriv(sigma, N)     # dS/dN
    dN_dX = np.transpose(W, (1,0))              # dN/dX
    
    return np.dot(dS_dN, dN_dX)

if __name__ == "__main__":
    
    np.random.seed(190203)

    X = np.random.randn(1,3)
    W = np.random.randn(3,1)
    
    np.set_printoptions(4)
    print(X)
    X[0,2] = X[0,2] + 0.01
    print("\n", X)
    print(matrix_function_backward_1(X, W, myMathFunctions.sigmoid))