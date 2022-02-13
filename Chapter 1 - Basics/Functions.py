import numpy as np 

def square(x: np.ndarray) -> np.ndarray:
    """Function squares each element of input ndarray"""
    return np.power(x, 2)

def leaky_relu(x: np.ndarray) -> np.ndarray:
    """Functions uses the "Leaky ReLU" for each element of input ndarray"""
    return np.maximum(0.2*x, x)

if __name__ == '__main__':
    a = np.array([[2, -1, 7, -4],[0, 1, -5, 0.5]])
    print(square(a))
    print(leaky_relu(a))