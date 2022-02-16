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
    