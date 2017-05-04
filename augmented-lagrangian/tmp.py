import numpy as np
from numpy import zeros, ones, zeros_like, ones_like, maximum, nan_to_num

n = 10
m = 20
W = np.random.randn(n, m)

def func(W):
    W_col_diff = 0.0
    for i in xrange(W.shape[0]-1):
        W_col_diff += ((W[:, i] - W[:, i+1])**2).sum()
    return W_col_diff

def grad(W):
    tilde_W = np.zeros_like(W)
    for i in xrange(W.shape[0]-1):
        tilde_W[:, i] -= 2.0 * (W[:, i] - W[:, i+1])
        tilde_W[:, i+1] += 2.0 * (W[:, i] - W[:, i+1])
    return tilde_W
    
print func(W)
print grad(W)

# fd check
eps = 0.01
fd_W = np.zeros_like(W)
for i in xrange(n):
    for j in xrange(n):
        W[i, j] += eps
        val1 = func(W)
        W[i, j] -= 2*eps
        val2 = func(W)
        W[i, j] += eps
        fd_W[i, j] = (val2-val1)/(2*eps)
print fd_W

print np.allclose(grad(W), fd_W, atol=0.01)
