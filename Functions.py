# This module will contain general functions used for the inversion of generative models

import math

import torch
# import autograd.numpy as np                 # autograd now implemented in JAX, can be replaced
# from autograd import grad                   # TODO: move this to pure pytorch
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

def spm_DEM_R(n, s):
    # function adapted and simplied from SPM, original description below; this is now in use as it is much faster for higher numbers of generalised coordinates

    # function [R,V] = spm_DEM_R(n,s,form)
    # returns the precision of the temporal derivatives of a Gaussian process
    # FORMAT [R,V] = spm_DEM_R(n,s,form)
    #__________________________________________________________________________
    # n    - truncation order
    # s    - temporal smoothness - s.d. of kernel {bins}
    # form - 'Gaussian', '1/f' [default: 'Gaussian']
    #
    #                         e[:] <- E*e(0)
    #                         e(0) -> D*e[:]
    #                 <e[:]*e[:]'> = R
    #                              = <E*e(0)*e(0)'*E'>
    #                              = E*V*E'
    #
    # R    - (n x n)     E*V*E: precision of n derivatives
    # V    - (n x n)     V:    covariance of n derivatives
    #__________________________________________________________________________
    # Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

    # Karl Friston
    # $Id: spm_DEM_R.m 4278 2011-03-31 11:48:00Z karl $

    # if no serial dependencies 
    #--------------------------------------------------------------------------
    if n == 0:
        R = torch.zeros(0,0)
        return R
    if s == 0:
        s = torch.exp(torch.tensor([-8]))



    # temporal correlations (assuming known form) - V
    #--------------------------------------------------------------------------
    # try, form; catch, form = 'Gaussian'; end

    # switch form

    #     case{'Gaussian'} # curvature: D^k(r) = cumprod(1 - k)/(sqrt(2)*s)^k
    #------------------------------------------------------------------
    k = torch.arange(0,n)
    x = torch.sqrt(torch.tensor([2.])) * s
    r = torch.zeros(2*n-1)
    r[2*k] = torch.cumprod((1 - 2*k), dim=0)/(x**(2*k))

    #     case{'1/f'}     # g(w) = exp(-x*w) => r(t) = sqrt(2/pi)*x/(x^2 + w^2)
    #         #------------------------------------------------------------------
    #         k          = [0:(n - 1)];
    #         x          = 8*s^2;
    #         r(1 + 2*k) = (-1).^k.*gamma(2*k + 1)./(x.^(2*k));
            
    #     otherwise
    #         errordlg('unknown autocorrelation')
    # end


    # create covariance matrix in generalised coordinates
    #==========================================================================
    V = torch.zeros(n, n)
    for i in range(n):
        V[i,:] = r[torch.arange(0,n) + i]
        r = -r

    # and precision - R
    #--------------------------------------------------------------------------
    R = V.inverse()

    return V









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

    # # return if ~q
    # #--------------------------------------------------------------------------
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
        y[:,i] = Y[:,k-1] @ E[i,:].t()         # TODO: Use torch.block_diag, i.e., noise = [[noise, zeros], [zeros, noise1gencoord]] ... or not? Everything in one column?
    
    z = y.t().flatten().t().unsqueeze(1)        # TODO: Look for more elegant solution

    # likely unnecessary, if we decide to stack all variables in one column

    # y = Y[:,k-1] @ E[0,:].t()
    # # embed
    # for i in range(1,n):
    #     y = torch.block_diag(y, Y[:,k-1] @ E[i,:].t())            

    return z




def spm_DEM_z(n, s, T, dt=1.):
    # see also https://www.kaggle.com/charel/learn-by-example-active-inference-noise/comments

    t = torch.arange(0, T, dt)                                          # autocorrelation lags
    K = torch.from_numpy(toeplitz(torch.exp(-t**2/(2*s**2))))           # convolution matrix
    K = torch.diag(1./torch.sqrt(torch.diag(K @ K.t()))) @ K
    # K  = diag(1./sqrt(diag(K*K')))*K                                    # spm

    noise = torch.randn(n, int(T/dt)) @ K
    return noise

def f(A, B_v, B_a, x, v, a):
    # TODO: generalise this to include nonlinear treatments
    try:
        return A @ x + B_v @ v + B_a @ a
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
    # D = kronecker(torch.eye(embeddings), torch.from_numpy(np.eye(dimension, k = shift)))    # TODO: torch does not support arbitrary diagonal shifts for the 'eye' function, numpy does
    offdiag = torch.diag(torch.ones(dimension), diagonal=shift)
    D = kronecker(torch.eye(embeddings), offdiag[:-shift,:-shift])    # TODO: find better workaround
    return D @ x