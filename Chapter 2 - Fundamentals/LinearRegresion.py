import numpy as np
from typing import Dict, Tuple
from matplotlib import pyplot as plt

def forward_linear_regression(X_batch: np.ndarray,
                              y_batch: np.ndarray,
                              weights: Dict[str, np.ndarray]
                              ) -> Tuple[float, Dict[str, np.ndarray]]:
    """Forward pass of linear regression"""
    
    assert X_batch.shape[0] == y_batch.shape[0], \
        "X and Y batch must be the same length"
        
    assert X_batch.shape[1] == weights['W'].shape[0], \
        """
        X_batch must has the same count of columns
        as weights has count of rows
        """
        
    assert weights['B'].shape[0] == weights['B'].shape[1] == 1 ,\
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

def to_2d_np(a: np.ndarray,
             type: str = "col") -> np.ndarray:
    """Function turns a 1D Tensor into 2D"""
    
    assert a.ndim == 1, \
        "Input tensor must be 1 dimensional"
        
    if type == "col":
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)
    
def permutate_data(X: np.ndarray, y: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Function permutates X and y along axis 0"""
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]

Batch = Tuple[np.ndarray, np.ndarray]

def generate_batch(X: np.ndarray,
                   y: np.ndarray,
                   start: int = 0,
                   batch_size: int = 10) -> Batch:
    """
    Function generates batch from X and y, given start position
    """
    assert X.ndim == y.ndim == 2, \
        "X and y must be 2 dimensional"
    
    if start + batch_size > X.shape[0]:
        batch_size = X.shape[0] - start
    
    X_batch, y_batch = X[start:start+batch_size], y[start:start+batch_size]

    return X_batch, y_batch

def init_weights(n_in: int) -> Dict[str, np.ndarray]:
    """
    Function initializes wieghts on first forward pass of model
    """
    weights: Dict [str, np.ndarray] = {}
    W = np.random.randn(n_in, 1)
    B = np.random.randn(1, 1)
    
    weights['W'] = W
    weights['B'] = B
    
    return weights

def train(X: np.ndarray,
          y: np.ndarray,
          n_iter: int = 1000,
          learning_rate: float = 0.01,
          batch_size: int = 100,
          return_losses: bool = False,
          return_weights: bool = False,
          seed: int = 1) -> None:
    """Function trains model for a certin number of epochs"""

    if seed:
        np.random.seed(seed)
    start = 0
    
    weights = init_weights(X.shape[1])
    
    # Permutate data
    X, y = permutate_data(X, y)
    
    if return_losses:
        losses = []
    
    for i in range(n_iter):     
        if start >= X.shape[0]:
            X, y = permutate_data(X, y)
            start = 0
        
        X_batch, y_batch = generate_batch(X, y, start, batch_size)
        start += batch_size
        
        loss, forward_info = forward_linear_regression(
            X_batch, y_batch, weights)
        
        if return_losses:
            losses.append(loss)
            
        loss_grads = loss_gradients(forward_info, weights)
        for key in weights.keys():
            weights[key] -=learning_rate * loss_grads[key]
            
    if return_weights:
        return losses, weights
        
    return None
        

if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
   
    np.set_printoptions(4)
    
    # Data preparation
    boston = load_boston()
    
    data = boston.data
    target = boston.target
    features = boston.feature_names
    
    s = StandardScaler()
    data = s.fit_transform(data)
    
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.3, random_state=80718)
    
    y_train = to_2d_np(y_train)
    
    train_info = train(X_train, y_train,
                       n_iter = 1000,
                       learning_rate = 0.001,
                       batch_size = 23,
                       return_losses = True,
                       return_weights = True,
                       seed = 180708)
    
    losses, weights = train_info
    # losses = train_info[0]
    # weights = train_info[1]
    print(len(losses), len(list(range(1000))))
    plt.plot(list(range(1000)), losses)
    plt.show()
    
    
    
    
    
    
    
    
    
    