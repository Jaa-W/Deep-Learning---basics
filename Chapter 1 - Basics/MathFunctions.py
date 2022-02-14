import numpy as np 
from typing import Callable
from matplotlib import pyplot as plt

def square(x: np.ndarray) -> np.ndarray:
    """Function squares each element of input ndarray"""
    return np.power(x, 2)

def leaky_relu(x: np.ndarray) -> np.ndarray:
    """Functions uses the "Leaky ReLU" for each element of input ndarray"""
    return np.maximum(0.2*x, x)

def deriv(func: Callable[[np.ndarray], np.ndarray],
          input_: np.ndarray,
          delta: float = 0.001) -> np.ndarray:
    """Functions calculates derivative of "func" function for each element
    from input_ ndarray"""
    return (func(input_ + delta) - func(input_ - delta)) / (2*delta)
    

if __name__ == '__main__':
    # a = np.array([[2, -1, 7, -4],[0, 1, -5, 0.5]])
    # print(square(a))
    # print(leaky_relu(a))
    
    x = np.arange(-3,3,0.01)
    
    plt.plot(x, square(x))
    plt.plot(x, deriv(square, x))
    plt.show()