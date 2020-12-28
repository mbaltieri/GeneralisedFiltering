# This module will contain general functions used for the inversion of generative models

import math

import torch
import autograd.numpy as np                 # autograd now implemented in JAX, can be replaced
from autograd import grad                   # TODO: move this to pure pytorch
from scipy.linalg import sqrtm, cholesky, toeplitz

torch.set_default_dtype(torch.float64)

### FUNCTIONS ###

def symsqrt(matrix):                # code taken from https://github.com/pytorch/pytorch/issues/25481, while we still wait for sqrtm
                                    # SVD > Cholemsky since the obtained "square root" is symmetric, and not lower or upper triangular
    """Compute the square root of a positive definite matrix."""
    _, s, v = matrix.svd()
    good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
    components = good.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.size(-1):
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]
    if unbalanced:
        s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)


def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

def sigmoid(x):
#    return x
    return torch.tanh(x)

def dsigmoid(x):
#    return x
    return 1 - torch.tanh(x)**2

# autocorrelation function (Gaussian)
def rho(h, phi):
    return torch.exp(-.5*h**2/phi)

# n-th derivative of rho function
def findDerivatives(orders):                   
    drho = [grad(rho,0)]
    for i in range(1,orders):
        drho.append(grad(drho[i-1]))

    return drho

def dnrho(h, phi, degree):
    return drho[degree](h, phi)

# TODO: This function is extremely slow at the moment due to the recurrent automatic 
# differentiation of the autocorrelation function (anything around 10 takes several second)
# Check: function [R,V] = spm_DEM_R(n,s,form) and spm_DEM_z
def temporalPrecisionMatrix(embedding_orders, phi):
    h = torch.tensor([[0.0]])                                           # lag
    derivative_order = (embedding_orders-1)*2+1
    rho_tilde = torch.zeros(derivative_order,)                          # array of autocorrelations
    S_inv = torch.zeros((embedding_orders,embedding_orders))            # temporal roughness
    S = torch.zeros((embedding_orders,embedding_orders))                # temporal smoothness

    drho = findDerivatives(derivative_order)
    rho_tilde[0] = rho(h, phi)
    for i in range(1, derivative_order):
        # rho_tilde[i] = dnrho(h, phi, i-1)
        rho_tilde[i] = drho[i-1](h, phi)

    for i in range(embedding_orders):
        for j in range(embedding_orders):
            S_inv[i, j] = np.power(-1, (i*j))*rho_tilde[i+j]            # roughness, this is to be multipled by the covariance matrix

    return S_inv

    # S = np.linalg.inv(S_inv)                                          # smoothness, this is to be multipled by the precision matrix

    # return torch.from_numpy(S)

def spm_DEM_embed(Y,n,t,dt=1,d=0):
    # function adapted and simplied from SPM, original description below

    # temporal embedding into derivatives
    # FORMAT [y] = spm_DEM_embed(Y,n,t,dt,d)
    #__________________________________________________________________________
    # Y    - (v x N) matrix of v time-series of length N
    # n    - order of temporal embedding
    # t    - time  {bins} at which to evaluate derivatives (starting at t = 1)
    # dt   - sampling interval {secs} [default = 1]
    # d    - delay (bins) for each row of Y
    #
    # y    - {n,1}(v x 1) temporal derivatives   y[:] <- E*Y(t)
    #==========================================================================
    # Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

    # Karl Friston
    # $Id: spm_DEM_embed.m 4663 2012-02-27 11:56:23Z karl $

    # get dimensions
    #--------------------------------------------------------------------------
    [q, N]  = list(Y.size())
    # y      = cell(n,1);
    # [y{:}] = deal(sparse(q,1));

    # % return if ~q
    # %--------------------------------------------------------------------------
    # if ~q, return, end

    # loop over channels
    
    # boundary conditions
    s      = torch.tensor([(t + 1 - d)/dt])
    k      = torch.arange(1,n+1) + torch.trunc(s - (n + 1)/2).long()
    x      = s - min(k) + 1
    i      = k < 1
    k      = k * ~i + i
    i      = k > N
    k      = k * ~i + i*N


    # Inverse embedding operator (T): cf, Taylor expansion Y(t) <- T*y[:]
    T = torch.zeros(n,n)
    for i in range(0,n):
        for j in range(0,n):
            T[i,j] = ((i + 1 - x) * dt)**j/math.factorial(j)

    # embedding operator: y[:] <- E*Y(t)
    E     = torch.inverse(T)

    y = torch.zeros(q, n)
    # embed
    for i in range(0,n):
        y[:,i]      = Y[:,k-1] @ E[i,:].t()
    return y




def spm_DEM_z(n, s, T, dt=1.):
    # see also https://www.kaggle.com/charel/learn-by-example-active-inference-noise/comments

    t = torch.arange(0, T, dt)                                          # autocorrelation lags
    K = torch.from_numpy(toeplitz(torch.exp(-t**2/(2*s**2))))           # convolution matrix
    K = torch.diag(1./torch.sqrt(torch.diag(K @ K.t()))) @ K
    # K  = diag(1./sqrt(diag(K*K')))*K                                    # spm

    noise = torch.randn(n, int(T/dt)) @ K
    return noise

def f(A, B, x, v):
    # TODO: generalise this to include nonlinear treatments
    try:
        return A @ x + B @ v
    except RuntimeError:
        print("Dimensions don't match!")
        return

def g(F, G, x, v):
    # TODO: generalise this to include nonlinear treatments
    try:
        return F @ x + G @ v
    except RuntimeError:
        print("Dimensions don't match!")
        return

def k_theta(D, eta_theta):                                                             # TODO: for simplicity we hereby assume that parameters are independent of other variables
    # TODO: generalise this to include nonlinear treatments
    try:
        return D @ eta_theta
    except RuntimeError:
        print("Dimensions don't match!")
        return
    
def k_gamma(E, eta_gamma):                                                             # TODO: for simplicity we hereby assume that hyperparameters are independent of other variables
    # TODO: generalise this to include nonlinear treatments
    try:
        return E @ eta_gamma
    except RuntimeError:
        print("Dimensions don't match!")
        return

def Diff(x, dimension, embeddings, shift=1):
    D = kronecker(torch.eye(embeddings), torch.from_numpy(np.eye(dimension, k = shift)))    # TODO: torch does not support arbitrary diagonal shifts for the 'eye' function, numpy does
    return D @ x
    

# def FreeEnergy(y, mu_x, mu_v, mu_pi_z, mu_pi_w, A_gm, B_gm, F_gm):
#     return .5 * (np.sum(np.dot(np.dot((y - np.dot(F_gm, mu_x)).transpose(), mu_pi_z), (y - np.dot(F_gm, mu_x)))) + \
#                 np.sum(np.dot(np.dot((mu_x[:, 1:] - f_gm(mu_x[:, :-1], mu_v[:, :-1], A_gm, B_gm)).transpose(), mu_pi_w), (mu_x[:, 1:] - f_gm(mu_x[:, :-1], mu_v[:, :-1], A_gm, B_gm)))) - \
#                 np.trace(np.log(mu_pi_z * mu_pi_w)))

# def F(y, x, v, theta, Pi_z, Pi_w, Pi_v, p_Pi_z, p_Pi_w):
#     return .5 * (torch.bmm(torch.bmm(torch.transpose(y - g(x, v, theta), 1, 2), Pi_z), y - g(x, theta, v)) + \
#                  torch.bmm(torch.bmm(torch.transpose(D(x, e_n) - f(x, v, theta), 1, 2), Pi_w), x[:,:,1:] - f(x, theta, v)) + \
#                  torch.bmm(torch.bmm(torch.transpose(v - h(x, theta, v), 1, 2), Pi_v), v - h(x, theta, v)) + \
#                  torch.bmm(torch.bmm(torch.transpose(Pi_z - p_1(x, theta, v), 1, 2), p_Pi_z), Pi_z - p_1(x, theta, v)) + \
#                  torch.bmm(torch.bmm(torch.transpose(Pi_w - p_2(x, theta, v), 1, 2), p_Pi_w), Pi_w - p_2(x, theta, v)) + \
#                  torch.log(torch.bmm(torch.bmm(torch.potrf(Pi_z).diag().prod(), torch.potrf(Pi_w).diag().prod()), torch.potrf(Pi_v).diag().prod())))

def F(A, F, B, C, G, H, D, E, n, r, p, h, e_n, e_r, e_p, e_h, y, x, v, eta_v, theta, eta_theta, gamma, eta_gamma, Pi_z, Pi_w, Pi_v, Pi_theta, Pi_gamma):
    eps_v = y - g(F, G, x, v)
    eps_x = Diff(x, n, e_n+1) - f(A, B, x, v)
    eps_eta = v - eta_v
    eps_theta = theta - k_theta(D, eta_theta)
    eps_gamma = gamma - k_gamma(E, eta_gamma)

    # print(eps_v)
    # print(eps_x)
    # print(eps_eta)
    # print(eps_theta)
    # print(eps_gamma)
    # print(torch.logdet(Pi_z))
    # print(torch.logdet(Pi_w))


    return .5 * (eps_v.t() @ Pi_z @ eps_v + eps_x.t() @ Pi_w @ eps_x + eps_eta.t() @ Pi_v @ eps_eta + \
                eps_theta.t() @ Pi_theta @ eps_theta + eps_gamma.t() @ Pi_gamma @ eps_gamma + \
                (n + r + p + h) * torch.log(2*torch.tensor([[math.pi]])) - torch.logdet(Pi_z) - torch.logdet(Pi_w) - \
                torch.logdet(Pi_theta) - torch.logdet(Pi_gamma)).sum()     # TODO: add more terms due to the variational Gaussian approximation?