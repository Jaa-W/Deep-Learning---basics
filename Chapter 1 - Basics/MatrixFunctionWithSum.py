import numpy as np
from typing import Callable
import MathFunctions as myMathFun

Array_Function = Callable[[np.ndarray], np.ndarray]

def matrix_function_forward_sum(X: np.ndarray,
                                W: np.ndarray,
                                sigma: Array_Function) -> float:
    """Function computing results of forward pass"""
    
    assert X.shape[1] == W.shape[0]
    
    N = np.dot(X, W)    # X x W
    S = sigma(N)
    L = np.sum(S)

    return L

def matrix_function_bakward_sum(X: np.ndarray,
                                W: np.ndarray,
                                sigma: Array_Function) -> float:
    """
    Function computing derivative of this function
    with respect to the first element.
    """
    
    assert X.shape[1] == W.shape[0]
    
    N = np.dot(X, W)    # X x W
    S = sigma(N)
    L = np.sum(S)
    
    dL_dS = np.ones_like(S)                 # dL/dS - same ones
    dS_dN = myMathFun.deriv(sigma, N) # dS/dN 
    dL_dN = dL_dS * dS_dN                   # dL/dN
    
    dN_dX = np.transpose(W, (1, 0))         # dN/dX
    
    dL_dX = np.dot(dL_dN, dN_dX)            # dL/dX
    
    return dL_dX    
    
if __name__ == "__main__":
    np.random.seed(190204)
    
    X = np.random.randn(3, 3)
    W = np.random.randn(3, 2)
    
    np.set_printoptions(4)
    print("X: \n", X)
    print("L:") 
    print(round(matrix_function_forward_sum(X, W, myMathFun.sigmoid),4))
    print("\ndL/dX: ")
    print(matrix_function_bakward_sum(X, W, myMathFun.sigmoid))
    
    X1 = X.copy()
    X1[0, 0] += 0.001
    print(round(
        (matrix_function_forward_sum(X1, W, myMathFun.sigmoid) - \
        matrix_function_forward_sum(X, W, myMathFun.sigmoid))/0.001,4))
    
    
    
    
    