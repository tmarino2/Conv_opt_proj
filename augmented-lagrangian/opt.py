import numpy as np
from numpy import zeros, zeros_like, maximum


def func(X, W, theta, H):
    """ function evaluation """
    estim = W.dot(np.diag(theta)).dot(H)
    return ((X - estim)**2).sum()


def fd_W(X, W, theta, H, eps=1.0):
    dW = zeros_like(W)
    for i in xrange(W.shape[0]):
        for j in xrange(W.shape[1]):
            W[i, j] += eps
            val1 = func(X, W, theta, H)
            W[i, j] -= 2*eps
            val2 = func(X, W, theta, H)
            W[i, j] += eps
            dW[i, j] = (val1-val2)/(2*eps)
    return dW


def fd_H(X, W, theta, H, eps=1.0):
    dH = zeros_like(H)
    for i in xrange(H.shape[0]):
        for j in xrange(H.shape[1]):
            H[i, j] += eps
            val1 = func(X, W, theta, H)
            H[i, j] -= 2*eps
            val2 = func(X, W, theta, H)
            H[i, j] += eps
            dH[i, j] = (val1-val2)/(2*eps)
    return dH


def fd_theta(X, W, theta, H, eps=1.0):
    dtheta = zeros_like(theta)
    for i in xrange(theta.shape[0]):
        theta[i] += eps
        val1 = func(X, W, theta, H)
        theta[i] -= 2*eps
        val2 = func(X, W, theta, H)
        theta[i] += eps
        dtheta[i] = (val1-val2)/(2*eps)
    return dtheta


def fd_H(X, W, theta, H, eps=1.0):
    dH = zeros_like(H)
    for i in xrange(H.shape[0]):
        for j in xrange(H.shape[1]):
            H[i, j] += eps
            val1 = func(X, W, theta, H)
            H[i, j] -= 2*eps
            val2 = func(X, W, theta, H)
            H[i, j] += eps
            dH[i, j] = (val1-val2)/(2*eps)
    return dH


def fd(eps=0.0001):
    n, m, r = 10, 20, 2
    for _ in xrange(20):
        X = maximum(np.random.randn(n, m), 0)
        W = maximum(np.random.randn(n, r), 0)
        H = maximum(np.random.randn(r, m), 0)
        theta = maximum(np.random.randn(r), 0)
        assert np.allclose(fd_W(X, W, theta, H, eps=eps), grad_W(X, W, theta, H), atol=0.001)
        assert np.allclose(fd_H(X, W, theta, H, eps=eps), grad_H(X, W, theta, H), atol=0.001)
        assert np.allclose(fd_theta(X, W, theta, H, eps=eps), grad_theta(X, W, theta, H), atol=0.001)

        
def grad_W(X, W, theta, H):
    estim = W.dot(np.diag(theta)).dot(H)
    return -2*(X - estim).dot((np.diag(theta).dot(H)).T)


def grad_H(X, W, theta, H):
    estim = W.dot(np.diag(theta)).dot(H)
    return ((W.dot(np.diag(theta))).T).dot(-2*(X - estim))


def grad_theta(X, W, theta, H):
    estim = W.dot(np.diag(theta)).dot(H)
    tmp = -2*(X - estim)
    return np.diag(np.dot(np.dot(W.T, tmp), H.T))

    
def opt(X, n, m, r, iters=200):

    X = maximum(np.random.randn(n, m), 0)
    W = maximum(np.random.randn(n, r), 0)
    H = maximum(np.random.randn(r, m), 0)
    theta = maximum(np.random.randn(r), 0)

    eta = 0.001
    
    for i in xrange(iters):


        for j in xrange(100):
            dW = grad_W(X, W, theta, H)
            W -= eta * dW
        for j in xrange(100):
            dH = grad_H(X, W, theta, H)
            H -= eta * dH
        for j in xrange(100):
            dtheta = grad_theta(X, W, theta, H)
            theta -= eta * dtheta
            # projection into the non-negative orthant
            theta = np.maximum(theta, 0)
            
        print func(X, W, theta, H)

            #dH = grad_H(X, W, theta, H)
        #    dtheta = grad_theta(X, W, theta, H)


if __name__ == "__main__":

    n, m, r = 10, 20, 10

    X = maximum(np.random.randn(n, m), 0)
    W = maximum(np.random.randn(n, r), 0)
    H = maximum(np.random.randn(r, m), 0)
    theta = maximum(np.random.randn(r), 0)


    #fd()


    opt(X, n, m, r)
