# This module will contain general functions used for the inversion of generative models

import math

import torch
# import autograd.numpy as np                 # autograd now implemented in JAX, can be replaced
# from autograd import grad                   # move this to pure pytorch
from scipy.linalg import sqrtm, cholesky, toeplitz

torch.set_default_dtype(torch.float64)

### FUNCTIONS ###

def symsqrt(matrix):                # code from https://github.com/pytorch/pytorch/issues/25481, while we still wait for sqrtm
                                    # SVD > (?) Cholemsky since the obtained "square root" is symmetric, and not lower or upper triangular
                                    # FIXME: find out if we should use Cholemsky instead https://www.gaussianwaves.com/2013/11/simulation-and-analysis-of-white-noise-in-matlab/
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


def kronecker(A, B):                # code from https://discuss.pytorch.org/t/kronecker-product/3919/9, torch.kron is almost ready (probably torch 1.8?)
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))


def Diff(x, dimension, embeddings, shift=1):
    # D = kronecker(torch.eye(embeddings), torch.from_numpy(np.eye(dimension, k = shift)))                  # torch does not yet support arbitrary shifts of the diagonal for the 'eye' function, numpy does, TODO: find better workaround?
    offdiag = torch.diag(torch.ones(dimension), diagonal=shift)
    if shift >= 0:
        D = kronecker(torch.eye(embeddings), offdiag[:-shift,:-shift])
    else:
        D = kronecker(torch.eye(embeddings), offdiag[:shift,:shift])
    return D @ x




def ff(x, u, a, A, B_u, B_a):
    # TODO: generalise this to include nonlinear treatments
    try:
        fx, fu, fa = ffSeparateComponents(x, u, a, A, B_u, B_a)
        return (fx + fu + fa)
    except RuntimeError:
        print("Dimensions don't match!")
        return

def ffSeparateComponents(x, u, a, A, B_u, B_a):
    # TODO: generalise this to include nonlinear treatments
    try:
        fx = A @ x
        fu = B_u @ u
        fa = B_a @ a
        return fx, fu, fa
    except RuntimeError:
        print("Dimensions don't match!")
        return

# def g(x, u, a, F, G):
#     # TODO: generalise this to include nonlinear treatments
#     try:
#         return F @ x + G @ u
#     except RuntimeError:
#         print("Dimensions don't match!")
#         return