import numpy as np
from numpy import zeros, ones, zeros_like, ones_like, maximum, nan_to_num
import scipy.io as sio
from scipy.optimize import fmin_l_bfgs_b as lbfgs

def func(X, W, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H):
    estim = W.dot(np.diag(theta)).dot(H)
    tmp = 1.0/X.shape[1] * ((X - estim)**2).sum() + L * np.abs(theta).sum() \
        + 1.0/2.0 * (np.maximum(-W + Mu_W/rho_W, 0)**2).sum() \
        + 1.0/2.0 * (np.maximum(-H + Mu_H/rho_H, 0)**2).sum() \
        + 1.0/2.0 * (np.maximum(-theta + Mu_theta/rho_theta, 0)**2).sum() 

    W_col_diff = 0.0
    for i in xrange(W.shape[0]-1):
        W_col_diff += ((W[:, i] - W[:, i+1])**2).sum()
        
    return tmp + eta * W_col_diff

def func_orig(X, W, theta, H, L, eta):
    estim = W.dot(np.diag(theta)).dot(H)

    W_col_diff = 0.0
    for i in xrange(W.shape[0]-1):
        W_col_diff += ((W[:, i] - W[:, i+1])**2).sum()
        
    return 1.0/X.shape[1] *  ((X - estim)**2).sum() + L * np.abs(theta).sum() + eta * W_col_diff


def fd_W(X, W, theta, H, L, eta, Mu_W,  Mu_theta, Mu_H, rho_W, rho_theta, rho_H, eps=1.0):
    dW = zeros_like(W)
    for i in xrange(W.shape[0]):
        for j in xrange(W.shape[1]):
            W[i, j] += eps
            val1 = func(X, W, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
            W[i, j] -= 2*eps
            val2 = func(X, W, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
            W[i, j] += eps
            dW[i, j] = (val1-val2)/(2*eps)
    return dW


def fd_H(X, W, theta, H, L, eta, Mu_W,  Mu_theta, Mu_H, rho_W, rho_theta, rho_H, eps=1.0):
    dH = zeros_like(H)
    for i in xrange(H.shape[0]):
        for j in xrange(H.shape[1]):
            H[i, j] += eps
            val1 = func(X, W, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
            H[i, j] -= 2*eps
            val2 = func(X, W, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
            H[i, j] += eps
            dH[i, j] = (val1-val2)/(2*eps)
    return dH


def fd_theta(X, W, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H, eps=1.0):
    dtheta = zeros_like(theta)
    for i in xrange(theta.shape[0]):
        theta[i] += eps
        val1 = func(X, W, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
        theta[i] -= 2*eps
        val2 = func(X, W, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
        theta[i] += eps
        dtheta[i] = (val1-val2)/(2*eps)
    return dtheta 


def fd_H(X, W, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H, eps=1.0):
    dH = zeros_like(H)
    for i in xrange(H.shape[0]):
        for j in xrange(H.shape[1]):
            H[i, j] += eps
            val1 = func(X, W, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
            H[i, j] -= 2*eps
            val2 = func(X, W, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
            H[i, j] += eps
            dH[i, j] = (val1-val2)/(2*eps)
    return dH


def fd(L, eta, eps=0.0001):
    n, m, r = 10, 20, 10
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

        assert np.allclose(fd_W(X, W, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H, eps=eps), grad_W(X, W, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H), atol=0.001)
        assert np.allclose(fd_H(X, W, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H, eps=eps), grad_H(X, W, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H), atol=0.001)
        assert np.allclose(fd_theta(X, W, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H, eps=eps), grad_theta(X, W, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H), atol=0.001)
        
def grad_W(X, W, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H):
    estim = W.dot(np.diag(theta)).dot(H)

    tilde_W = np.zeros_like(W)
    for i in xrange(W.shape[0]-1):
        tilde_W[:, i] += 2.0 * (W[:, i] - W[:, i+1])
        tilde_W[:, i+1] -= 2.0 * (W[:, i] - W[:, i+1])
    
    return -2.0/X.shape[1] *(X - estim).dot((np.diag(theta).dot(H)).T) \
                    - np.maximum(-W + Mu_W/rho_W, 0) + eta * tilde_W

def grad_H(X, W, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H):
    estim = W.dot(np.diag(theta)).dot(H)
    return 1.0/X.shape[1] * ((W.dot(np.diag(theta))).T).dot(-2*(X - estim)) \
                      - np.maximum(-H + Mu_H/rho_H, 0)

def grad_theta(X, W, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H):
    estim = W.dot(np.diag(theta)).dot(H)
    tmp = -2*(X - estim)
    return 1.0/X.shape[1] * np.diag(np.dot(np.dot(W.T, tmp), H.T)) + L * np.sign(theta) \
        - np.maximum(-theta + Mu_theta/rho_theta, 0)

    
def opt(X, n, m, r, L, eta, iters=2000, fhandle=None, fudge=0.01):

    W = maximum(np.random.randn(n, r), 0)
    H = maximum(np.random.randn(r, m), 0)
    theta = maximum(np.random.randn(r), 0)
    
    mu_max = 1000.

    Mu_W = maximum(np.random.randn(n, r), 0) + fudge
    rho_W = ones_like(Mu_W)
    Mu_H = maximum(np.random.randn(r, m), 0) + fudge
    rho_H = ones_like(Mu_H)
    Mu_theta = maximum(np.random.randn(r), 0) + fudge
    rho_theta = ones_like(Mu_theta)
    
    for k in xrange(iters):
        
        # iterate on W
        for i in xrange(50):
            params = W.reshape(-1)
            def f(params):
                W_new = np.resize(params, W.shape)
                np.resize(params, W.shape)
                return func(X, W_new, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
            
            def g(params):
                W_new = np.resize(params, W.shape)
                return np.resize(grad_W(X, W_new, theta, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H), params.shape)
        
            opt, _, _ = lbfgs(f, params, fprime=g, disp=0, maxiter=10)
            W = opt.reshape(W.shape)
            Mu_W += rho_W * np.maximum(0, -W)
            V_W = np.minimum(W, Mu_W/rho_W)
            rho_W *= 1.01
            Mu_W = np.minimum(Mu_W, mu_max)

            string = "innerW {0} {1}: {2}".format(*(i, k, func_orig(X, W, theta, H, L, eta)))
            print string
            if fhandle is not None:
                fhandle.write(string+"\n")

        W = np.maximum(0, W)

        string = "W {0}: {1}".format(*(k, func_orig(X, W, theta, H, L, eta)))
        print string
        if fhandle is not None:
            fhandle.write(string+"\n")


        # iterate on H
        for i in xrange(50):
            params = H.reshape(-1)
            def f(params):
                H_new = np.resize(params, H.shape)
                np.resize(params, H.shape)
                return func(X, W, theta, H_new, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
        
            def g(params):
                H_new = np.resize(params, H.shape)
                return np.resize(grad_H(X, W, theta, H_new, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H), params.shape)

            opt, _, _ = lbfgs(f, params, fprime=g, disp=0, maxiter=10)
            H = opt.reshape(H.shape)
            Mu_H += rho_H * np.maximum(0, -H)
            V_H = np.minimum(H, Mu_H/rho_H)
            rho_H *= 1.01
            Mu_H = np.minimum(Mu_H, mu_max)

            string = "innerH {0} {1}: {2}".format(*(i, k, func_orig(X, W, theta, H, L, eta)))
            print string
            if fhandle is not None:
                fhandle.write(string+"\n")
            
        H = np.maximum(0, H)

        string = "H {0}: {1}".format(*(k, func_orig(X, W, theta, H, L, eta)))
        print string
        if fhandle is not None:
            fhandle.write(string+"\n")

        # iterate on theta
        for i in xrange(50):
            params = theta.reshape(-1)
            def f(params):
                theta_new = np.resize(params, theta.shape)
                np.resize(params, theta.shape)
                return func(X, W, theta_new, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H)
            
            def g(params):
                theta_new = np.resize(params, theta.shape)
                return np.resize(grad_theta(X, W, theta_new, H, L, eta, Mu_W, Mu_theta, Mu_H, rho_W, rho_theta, rho_H), params.shape)
        
            opt, _, _ = lbfgs(f, params, fprime=g, disp=0, maxiter=100)
            theta = opt.reshape(theta.shape)
        
            Mu_theta += rho_theta * np.maximum(0, -theta)
            V_theta = np.minimum(theta, Mu_theta/rho_theta)
            rho_theta *= 1.01
            Mu_theta = np.minimum(Mu_theta, mu_max)

            string = "innertheta {0} {1}: {2}".format(*(i, k, func_orig(X, W, theta, H, L, eta)))
            print string
            if fhandle is not None:
                fhandle.write(string+"\n")
            
        theta = np.maximum(0, theta)
        
        string = "theta {0}: {1}".format(*(k, func_orig(X, W, theta, H, L, eta)))
        print string
        if fhandle is not None:
            fhandle.write(string+"\n")
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--X', type=str)
    parser.add_argument('--log', type=str)
    parser.add_argument('--L', type=float)
    parser.add_argument('--eta', type=float)
    args = parser.parse_args()
    
    M = sio.loadmat(args.X)
    X = M['X']
    n, m, r = X.shape[0], X.shape[1], 40

    W = maximum(np.random.randn(n, r), 0)
    H = maximum(np.random.randn(r, m), 0)
    theta = maximum(np.random.randn(r), 0)

    L = args.L
    eta = args.eta

    # UNIT TEST
    # fd(L, eta)a
    # exit(0)

    with open(args.log, 'wb') as f:
        f.write("lambda={0}\n".format(L))
        f.write("eta={0}\n".format(eta))
        
        opt(X, n, m, r, L, eta, iters=5000, fhandle=f)

    # TO RUN
    # python opt.py --X ../data1.mat --log dump --L 0.01 --eta 0.01
