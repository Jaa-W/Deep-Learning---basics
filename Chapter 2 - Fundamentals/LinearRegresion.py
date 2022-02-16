import numpy as np
from typing import Dict, Tuple

def forward_linear_regression(X_batch: np.ndarray,
                              y_batch: np.ndarray,
                              weights: Dict[str, np.ndarray]
                              ) -> Tuple[float, Dict[str, np.ndarray]]:
    """Forward pass of linear regression"""
    
    assert X_batch.shape[0] == y_batch.shape[0], \
        "X and Y batch must be the same length"
        
    assert X_batch.shape[1] == weights.shape[0], \
        """
        X_batch must has the same count of columns
        as weights has count of rows
        """
        
    assert weights['B'].shape[0] == X_batch['B'].shape[1] == 1 ,\
        "B array must be shape of 1x1"
        
    N = np.dot(X_batch,weights['W'])
    P = N + weights['B']
    
    loss = np.mean(np.power(y_batch -P, 2))
    
    forward_info: Dict[str, np.ndarray] = {}
    forward_info['X'] = X_batch
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['y'] = y_batch
    
    return loss, forward_info
    
def loss_gradients(forward_info: Dict[str, np.ndarray],
                   weights: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Function compute dL/dW i dL/dB for step-by-step linear regression model
    """
    
    batch_size = forward_info['X'].shape[0]
    
    dL_dP = -2 * (forward_info['y'] - forward_info['P'])    # dL/dP
    
    dP_dN = np.ones_like(forward_info['N'])                 # dP/dN
    dP_dB = np.ones_like(weights['B'])                      # dP/dB
    
    dL_dN = dL_dP * dP_dN                                   # dL/dN
    
    dN_dW = np.transpose(forward_info['X'], (1,0))          # dN/dW
    
    # dN_dW must be at LEFT side
    dL_dW = np.dot(dN_dW, dL_dN)                            # dL/dW
    dL_dB = (dL_dP * dP_dB).sum(axis = 0)                   # dL/dB
    
    loss_gradients: Dict[str, np.ndarray] = {}
    loss_gradients['W'] = dL_dW
    loss_gradients['B'] = dL_dB
    
    return loss_gradients

if __name__ == "__main__":
    
    np.set_printoptions(4)
    
    
    