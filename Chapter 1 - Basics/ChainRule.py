import numpy as np
from matplotlib import pyplot as plt
from typing import Callable, List
import MathFunctions as myMathFunctions

Array_Function = Callable[[np.ndarray], np.ndarray]
Chain = List[Array_Function]

def chain_length_2(chain: Chain,
                   x: np.ndarray) -> np.ndarray:
    """Function processing 2 funcs in chain"""
    
    assert len(chain) == 2, \
        "Chain length must be 2"
    
    f1 = chain[0]
    f2 = chain[1]
    
    return f2(f1(x))

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Function uses sigmoid function for each element of input array""" 
    return 1 / (1 + np.exp(-x))

def chain_deriv_2(chain: Chain,
                  input_range: np.ndarray) -> np.ndarray:
    """Function uses chain rule to compute derivates"""
    assert len(chain) == 2,\
        "chain must be length of 2"
        
    assert input_range.ndim == 1,\
        "input_range must have 1 dimension"
        
    f1 = chain[0]
    f2 = chain[1]
    
    f1_of_x = f1(input_range)                           # df1/dx
    df1_dx = myMathFunctions.deriv(f1, input_range)     # df1/du
    
    df2_du = myMathFunctions.deriv(f2, f1_of_x) # df2/du(f1(x))
    
    return df1_dx* df2_du

def plot_chain(ax: plt.Axes,
               chain: Chain,
               input_range: np.ndarray) -> None:
    """
    Plots chain function 
    ax: matplotlib Subplot for plotting
    """
    
    assert input_range.ndim == 1, \
        "input_range dimension must be 1"
    
    output_range = chain_length_2(chain, input_range)
    ax.plot(input_range, output_range)
    
def plot_chain_deriv(ax: plt.Axes,
                     chain: Chain,
                     input_range: np.ndarray) -> None:
    """
    Plots result of chain rule function
    ax: matplotlib subplot for plotting
    """
    
    assert input_range.ndim == 1, \
        "input_range dimension must be 1"
        
    output_range = chain_deriv_2(chain, input_range)
    ax.plot(input_range, output_range)
    
if __name__ == "__main__" :
    PLOT_RANGE = np.arange(-3, 3, 0.01)
    
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
    
    chain_1 = [myMathFunctions.square, sigmoid]
    chain_2 = [sigmoid, myMathFunctions.square]
    
    plot_chain(ax[0], chain_1, PLOT_RANGE)  
    plot_chain_deriv(ax[0], chain_1, PLOT_RANGE)
    
    plot_chain(ax[1], chain_2, PLOT_RANGE)  
    plot_chain_deriv(ax[1], chain_2, PLOT_RANGE)
    
    plt.show()
    
      