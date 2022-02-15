import numpy as np
from matplotlib import pyplot as plt
import ChainRule as myChainRule
import MathFunctions as myMathFun

def chain_length_3(chain: myChainRule.Chain,
                   x: np.ndarray) -> np.ndarray:
    """Function processing 2 funcs in chain"""
    
    assert len(chain) == 3, \
        "Chain length must be 2"
    
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]
    
    return f3(f2(f1(x)))

def chain_deriv_3(chain: myChainRule.Chain,
                  input_range: np.ndarray) -> np.ndarray:
    """Function uses chain rule to caluclate derivative of 3 nested functions"""
    
    assert len(chain) == 3, \
        "Chain must have length 3"
        
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]
    
    f1_of_x = f1(input_range)       # f1(x)
    f2_of_x = f2(f1_of_x)           # f2(f1(x))

    df3_du = myMathFun.deriv(f3, f2_of_x)     # df3/du
    df2_du = myMathFun.deriv(f2, f1_of_x)     # df2/du 
    df1_du = myMathFun.deriv(f1, input_range) # df1/du
    
    return df3_du * df2_du * df1_du

def plot_chain(chain: myChainRule.Chain,
               input_range: np.ndarray) -> None:
    """
    Plots chain function 
    ax: matplotlib Subplot for plotting
    """
    
    assert input_range.ndim == 1, \
        "input_range dimension must be 1"
    
    output_range = chain_length_3(chain, input_range)
    plt.plot(input_range, output_range)
    
def plot_chain_deriv(chain: myChainRule.Chain,
                     input_range: np.ndarray) -> None:
    """
    Plots result of chain rule function
    ax: matplotlib subplot for plotting
    """
    
    assert input_range.ndim == 1, \
        "input_range dimension must be 1"
        
    output_range = chain_deriv_3(chain, input_range)
    plt.plot(input_range, output_range)

if __name__ == "__main__":
    
    PLOT_RANGE = np.arange(-3, 3, 0.01)
    
    chain = [
        myMathFun.leaky_relu, 
        myMathFun.sigmoid, 
        myMathFun.square
        ]
    
    plot_chain(chain, PLOT_RANGE)
    plot_chain_deriv(chain, PLOT_RANGE)
    
    plt.show()