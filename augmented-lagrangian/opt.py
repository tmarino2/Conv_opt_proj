import numpy as np
from numpy import zeros, ones, zeros_like, ones_like, maximum


def func(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H):
    estim = W.dot(np.diag(theta)).dot(H)
    return ((X - estim)**2).sum() + L * np.abs(theta).sum() \
        + 1.0/2.0 * (np.maximum(W + Mu_W, 0)**2).sum() \
        + 1.0/2.0 * (np.maximum(H + Mu_H, 0)**2).sum() \
        + 1.0/2.0 * (np.maximum(theta + Mu_theta, 0)**2).sum() 


def fd_W(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H, eps=1.0):
    dW = zeros_like(W)
    for i in xrange(W.shape[0]):
        for j in xrange(W.shape[1]):
            W[i, j] += eps
            val1 = func(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H)
            W[i, j] -= 2*eps
            val2 = func(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H)
            W[i, j] += eps
            dW[i, j] = (val1-val2)/(2*eps)
    return dW


def fd_H(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H, eps=1.0):
    dH = zeros_like(H)
    for i in xrange(H.shape[0]):
        for j in xrange(H.shape[1]):
            H[i, j] += eps
            val1 = func(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H)
            H[i, j] -= 2*eps
            val2 = func(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H)
            H[i, j] += eps
            dH[i, j] = (val1-val2)/(2*eps)
    return dH


def fd_theta(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H, eps=1.0):
    dtheta = zeros_like(theta)
    for i in xrange(theta.shape[0]):
        theta[i] += eps
        val1 = func(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H)
        theta[i] -= 2*eps
        val2 = func(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H)
        theta[i] += eps
        dtheta[i] = (val1-val2)/(2*eps)
    return dtheta 


def fd_H(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H, eps=1.0):
    dH = zeros_like(H)
    for i in xrange(H.shape[0]):
        for j in xrange(H.shape[1]):
            H[i, j] += eps
            val1 = func(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H)
            H[i, j] -= 2*eps
            val2 = func(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H)
            H[i, j] += eps
            dH[i, j] = (val1-val2)/(2*eps)
    return dH


def fd(L, eps=0.0001):
    n, m, r = 10, 20, 2
    for _ in xrange(20):
        X = maximum(np.random.randn(n, m), 0)
        
        W = maximum(np.random.randn(n, r), 0)
        H = maximum(np.random.randn(r, m), 0)
        theta = maximum(np.random.randn(r), 0)

        Mu_W = maximum(np.random.randn(n, r), 0) 
        Mu_H = maximum(np.random.randn(r, m), 0) 
        Mu_theta = maximum(np.random.randn(r), 0)
        
        assert np.allclose(fd_W(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H, eps=eps), grad_W(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H), atol=0.001)
        assert np.allclose(fd_H(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H, eps=eps), grad_H(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H), atol=0.001)
        assert np.allclose(fd_theta(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H, eps=eps), grad_theta(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H), atol=0.001)

        
def grad_W(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H):
    estim = W.dot(np.diag(theta)).dot(H)
    return -2*(X - estim).dot((np.diag(theta).dot(H)).T) \
        + np.maximum(W + Mu_W, 0)


def grad_H(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H):
    estim = W.dot(np.diag(theta)).dot(H)
    return ((W.dot(np.diag(theta))).T).dot(-2*(X - estim)) \
        + np.maximum(H + Mu_H, 0)


def grad_theta(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H):
    estim = W.dot(np.diag(theta)).dot(H)
    tmp = -2*(X - estim)
    return np.diag(np.dot(np.dot(W.T, tmp), H.T)) + L * np.sign(theta) \
        + np.maximum(theta + Mu_theta, 0)

    
def opt(X, n, m, r, iters=2000):

    X = maximum(np.random.randn(n, m), 0)
    W = maximum(np.random.randn(n, r), 0)
    H = maximum(np.random.randn(r, m), 0)
    theta = maximum(np.random.randn(r), 0)

    Mu_W = ones_like(W)
    Mu_H = ones_like(H)
    Mu_theta = ones_like(theta)

    rho = 0.1
    
    eta = 0.001
    
    for i in xrange(iters):

        for j in xrange(100):
            dW = grad_W(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H)
            W -= eta * dW
        for j in xrange(100):
            dH = grad_H(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H)
            H -= eta * dH
        for j in xrange(100):
            dtheta = grad_theta(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H)
            theta -= eta * dtheta
            # projection into the non-negative orthant
            # theta = np.maximum(theta, 0)

        print func(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H)

        #Mu_W += rho * np.maximum(0, W)
        #Mu_H += rho * np.maximum(0, H)
        #Mu_theta += rho * np.maximum(0, theta)

        #dH = grad_H(X, W, theta, H)
        #    dtheta = grad_theta(X, W, theta, H)

        print "W min", W.min()
        #print "H min", Mu_H.min()
        #print "theta min", Mu_theta.min()

if __name__ == "__main__":

    n, m, r = 10, 20, 10

    X = maximum(np.random.randn(n, m), 0)
    W = maximum(np.random.randn(n, r), 0)
    H = maximum(np.random.randn(r, m), 0)
    theta = maximum(np.random.randn(r), 0)


    L = 0.1
    #fd(L)

    opt(X, n, m, r)
