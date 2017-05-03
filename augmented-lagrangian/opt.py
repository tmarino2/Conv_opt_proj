import numpy as np
from numpy import zeros, ones, zeros_like, ones_like, maximum, nan_to_num
import scipy.io as sio
from scipy.optimize import fmin_l_bfgs_b as lbfgs

def func(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H):
    estim = W.dot(np.diag(theta)).dot(H)
    return 1.0/X.shape[1] * ((X - estim)**2).sum() + L * np.abs(theta).sum() \
        + 1.0/2.0 * (np.maximum(-W + Mu_W/rho_W, 0)**2).sum() \
        + 1.0/2.0 * (np.maximum(-H + Mu_H/rho_H, 0)**2).sum() \
        + 1.0/2.0 * (np.maximum(-theta + Mu_theta/rho_theta, 0)**2).sum() 


def func_orig(X, W, theta, H):
    estim = W.dot(np.diag(theta)).dot(H)
    return 1.0/X.shape[1] *  ((X - estim)**2).sum() + L * np.abs(theta).sum() + L * np.abs(theta).sum()


def fd_W(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H, rho_W, rho_theta, rho_H, eps=1.0):
    dW = zeros_like(W)
    for i in xrange(W.shape[0]):
        for j in xrange(W.shape[1]):
            W[i, j] += eps
            val1 = func(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
            W[i, j] -= 2*eps
            val2 = func(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
            W[i, j] += eps
            dW[i, j] = (val1-val2)/(2*eps)
    return dW


def fd_H(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H, rho_W, rho_theta, rho_H, eps=1.0):
    dH = zeros_like(H)
    for i in xrange(H.shape[0]):
        for j in xrange(H.shape[1]):
            H[i, j] += eps
            val1 = func(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
            H[i, j] -= 2*eps
            val2 = func(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
            H[i, j] += eps
            dH[i, j] = (val1-val2)/(2*eps)
    return dH


def fd_theta(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H, rho_W, rho_theta, rho_H, eps=1.0):
    dtheta = zeros_like(theta)
    for i in xrange(theta.shape[0]):
        theta[i] += eps
        val1 = func(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
        theta[i] -= 2*eps
        val2 = func(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
        theta[i] += eps
        dtheta[i] = (val1-val2)/(2*eps)
    return dtheta 


def fd_H(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H, rho_W, rho_theta, rho_H, eps=1.0):
    dH = zeros_like(H)
    for i in xrange(H.shape[0]):
        for j in xrange(H.shape[1]):
            H[i, j] += eps
            val1 = func(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
            H[i, j] -= 2*eps
            val2 = func(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
            H[i, j] += eps
            dH[i, j] = (val1-val2)/(2*eps)
    return dH


def fd(L, eps=0.0001):
    n, m, r = 10, 20, 5
    for _ in xrange(20):
        X = maximum(np.random.randn(n, m), 0)
        
        W = maximum(np.random.randn(n, r), 0)
        H = maximum(np.random.randn(r, m), 0)
        theta = maximum(np.random.randn(r), 0)

        fudge = 0.1
        Mu_W = maximum(np.random.randn(n, r), 0) + fudge
        Mu_H = maximum(np.random.randn(r, m), 0) + fudge
        Mu_theta = maximum(np.random.randn(r), 0) + fudge

        rho_W = maximum(np.random.randn(n, r), 0) + fudge
        rho_H = maximum(np.random.randn(r, m), 0) + fudge
        rho_theta = maximum(np.random.randn(r), 0) + fudge

        assert np.allclose(fd_W(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H, eps=eps), grad_W(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H), atol=0.001)
        assert np.allclose(fd_H(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H, eps=eps), grad_H(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H), atol=0.001)
        assert np.allclose(fd_theta(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H, eps=eps), grad_theta(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H), atol=0.001)

        
def grad_W(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H):
    estim = W.dot(np.diag(theta)).dot(H)
    return -2.0/X.shape[1] *(X - estim).dot((np.diag(theta).dot(H)).T) \
                    - np.maximum(-W + Mu_W/rho_W, 0)


def grad_H(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H):
    estim = W.dot(np.diag(theta)).dot(H)
    return 1.0/X.shape[1] * ((W.dot(np.diag(theta))).T).dot(-2*(X - estim)) \
                      - np.maximum(-H + Mu_H/rho_H, 0)

def grad_theta(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H):
    estim = W.dot(np.diag(theta)).dot(H)
    tmp = -2*(X - estim)
    return 1.0/X.shape[1] * np.diag(np.dot(np.dot(W.T, tmp), H.T)) + L * np.sign(theta) \
        - np.maximum(-theta + Mu_theta/rho_theta, 0)

    
def opt(X, n, m, r, L, iters=2000, fudge=0.01):

    X = maximum(np.random.randn(n, m), 0)
    W = maximum(np.random.randn(n, r), 0)
    H = maximum(np.random.randn(r, m), 0)
    theta = maximum(np.random.randn(r), 0)

    mu_max = 100.
    
    Mu_W = maximum(np.random.randn(n, r), 0) + fudge
    Mu_H = maximum(np.random.randn(r, m), 0) + fudge
    Mu_theta = maximum(np.random.randn(r), 0) + fudge

    Mu_W = zeros_like(Mu_W)
    Mu_H = zeros_like(Mu_H)
    Mu_theta = zeros_like(Mu_theta)

    
    # rho_W = maximum(np.random.randn(n, r), 0) + fudge
    # rho_H = maximum(np.random.randn(r, m), 0) + fudge
    # rho_theta = maximum(np.random.randn(r), 0) + fudge
    rho_W = ones_like(Mu_W)
    rho_H = ones_like(Mu_H)
    rho_theta = ones_like(Mu_theta)
    
    eta_theta = 0.00001
    eta_W = 0.0005
    eta_H = 0.0005
    
    for k in xrange(iters):

        # STEP 1
        
        # iterate on W
        params = W.reshape(-1)
        def f(params):
            W_new = np.resize(params, W.shape)
            np.resize(params, W.shape)
            return func(X, W_new, theta, H, L, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
            
        def g(params):
            W_new = np.resize(params, W.shape)
            return np.resize(grad_W(X, W_new, theta, H, L, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H), params.shape)

        opt, _, _ = lbfgs(f, params, fprime=g, disp=0, maxiter=100)
        W = opt.reshape(W.shape)
        #W = np.maximum(0, W)
        
        # iterate on H
        params = H.reshape(-1)
        def f(params):
            H_new = np.resize(params, H.shape)
            np.resize(params, H.shape)
            return func(X, W, theta, H_new, L, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
            
        def g(params):
            H_new = np.resize(params, H.shape)
            return np.resize(grad_H(X, W, theta, H_new, L, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H), params.shape)

        opt, _, _ = lbfgs(f, params, fprime=g, disp=0, maxiter=100)
        H = opt.reshape(H.shape)
        #H = np.maximum(0, H)
        
        # iterate on theta
        params = theta.reshape(-1)
        def f(params):
            theta_new = np.resize(params, theta.shape)
            np.resize(params, theta.shape)
            return func(X, W, theta_new, H, L, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
            
        def g(params):
            theta_new = np.resize(params, theta.shape)
            return np.resize(grad_theta(X, W, theta_new, H, L, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H), params.shape)
        
        opt, _, _ = lbfgs(f, params, fprime=g, disp=0, maxiter=100)
        theta = opt.reshape(theta.shape)
        #theta = np.maximum(0, theta)


        #print func(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
        #continue
    
        
        # # STEP 1
        # for j in xrange(100):
        #     dW = grad_W(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
        #     W -= eta_W * dW
        #     #W = np.maximum(W, 0)
            
        # for j in xrange(100):
        #     dH = grad_H(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
        #     H -= eta_H * dH
        #     #H = np.maximum(H, 0)
            
        # for j in xrange(100):
        #     dtheta = grad_theta(X, W, theta, H, L, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
        #     theta -= eta_theta * dtheta
        #     # projection into the non-negative orthant
        #     # this is sort of a hack, but it keeps it stable
        #     #theta = np.maximum(theta, 0)

        print func(X, W, theta, H, L, Mu_W,  Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
        print func_orig(X, W, theta, H)
        # continue
    
        # STEP 2
        Mu_W += rho_W * np.maximum(0, -W)
        Mu_H += rho_H * np.maximum(0, -H)
        Mu_theta += rho_theta * np.maximum(0, -theta)

        # STEP 3
        V_W = np.minimum(W, Mu_W/rho_W)
        V_H = np.minimum(H, Mu_H/rho_H)
        V_theta = np.minimum(theta, Mu_theta/rho_theta)

        # condition missing (vacuous?)
        rho_W *= 1.01
        rho_H *= 1.01
        rho_theta *= 1.01

        # STEP 4
        Mu_W = np.minimum(Mu_W, mu_max)
        Mu_H = np.minimum(Mu_H, mu_max)
        Mu_theta = np.minimum(Mu_theta, mu_max)

        # STEP 5
        # increment
        
        print "Mu_W", Mu_W.min(), Mu_W.max()
        print "Mu_H", Mu_H.min(), Mu_H.max()
        print "Mu_theta", Mu_theta.min(), Mu_theta.max()

        print "W:", W.min(), W.max()
        print "H:", H.min(), H.max()
        print "theta:", theta.min(), theta.max()

if __name__ == "__main__":
    import sys
    
    M = sio.loadmat(sys.argv[1])
    X = M['X']
    n, m, r = X.shape[0], X.shape[1], 40

    #n, m, r = 10, 20, 10
    X = maximum(np.random.randn(n, m), 0)
    W = maximum(np.random.randn(n, r), 0)
    H = maximum(np.random.randn(r, m), 0)
    theta = maximum(np.random.randn(r), 0)


    #print W.shape
    #print theta.shape

    #exit(0)

    L = 0.001
    #fd(L)
    #exit(0)
    opt(X, n, m, r, L)
