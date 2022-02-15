import numpy as np
from typing import Callable
from matplotlib import pyplot as plt
import MathFunctions as myMathFun

Array_Function = Callable[[np.ndarray], np.ndarray]

def multiple_inputs_add(x: np.ndarray,
                        y: np.ndarray,
                        sigma: Array_Function) -> float:
    """Function takes multiple inputs and add them"""
    
    assert x.shape == y.shape, \
        "x and y must have the same shape"
        
    a = x + y
    return sigma(a)

def multiple_inputs_add_backward(x: np.ndarray,
                                 y: np.ndarray,
                                 sigma: Array_Function) -> float:
    """Function calculates derivative with respect to both inputs"""
    
    a = x + y
    
    ds_da = myMathFun.deriv(sigma, a)
    
    da_dx, da_dy = 1, 1
    return ds_da * da_dx, ds_da *da_dy

if __name__ == "__main__":
    x = np.arange(-2,2,0.1)
    y = np.arange(4,-4,-0.2)

    plt.plot(multiple_inputs_add(x, y, myMathFun.square))
    # plt.plot(multiple_inputs_add_backward(x, y, myMathFunctions.square))
    plt.show()