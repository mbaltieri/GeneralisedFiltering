# This module will contain general functions used for the inversion of generative models

import math

import torch
# import autograd.numpy as np                 # autograd now implemented in JAX, can be replaced
# from autograd import grad                   # move this to pure pytorch
from scipy.linalg import sqrtm, cholesky, toeplitz

torch.set_default_dtype(torch.float64)

### FUNCTIONS ###

def symsqrt(matrix):                # code from https://github.com/pytorch/pytorch/issues/25481, while we still wait for sqrtm
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


def kronecker(A, B):                # code from https://discuss.pytorch.org/t/kronecker-product/3919/9, torch.kron is almost ready (probably torch 1.8?)
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))


def Diff(x, dimension, embeddings, shift=1):
    # D = kronecker(torch.eye(embeddings), torch.from_numpy(np.eye(dimension, k = shift)))    # TODO: torch does not support arbitrary diagonal shifts for the 'eye' function, numpy does
    offdiag = torch.diag(torch.ones(dimension), diagonal=shift)
    if shift >= 0:
        D = kronecker(torch.eye(embeddings), offdiag[:-shift,:-shift])    # TODO: find better workaround
    else:
        D = kronecker(torch.eye(embeddings), offdiag[:shift,:shift])    # TODO: find better workaround
    return D @ x


class freeEnergy():
    def init()
def prediction_errors(self, i):
        self.eps_v = self.y[i,:] - self.g(i)
        self.eps_x = Diff(self.x[i,:], self.n, self.e_sim+1) - self.f(i)
        self.eps_eta = self.u[i,:] - self.eta_u[i,:]
        self.eps_theta = self.theta - self.k_theta()
        self.eps_gamma = self.gamma - self.k_gamma()

        # weighted prediction errors
        self.xi_v = self.Pi_z @ self.eps_v
        self.xi_v.retain_grad()
        self.xi_x = self.Pi_w @ self.eps_x
        self.xi_eta = self.Pi_v @ self.eps_eta
        self.xi_theta = self.Pi_theta @ self.eps_theta
        self.xi_gamma = self.Pi_gamma @ self.eps_gamma

        self.saveHistoryPredictionErrors(i)

    def saveHistoryPredictionErrors(self, i):
        self.eps_v_history[i, :] = self.eps_v
        self.eps_x_history[i, :] = self.eps_x
        self.eps_eta_history[i, :] = self.eps_eta
        self.eps_theta_history[i, :] = self.eps_theta
        self.eps_gamma_history[i, :] = self.eps_gamma
    
        self.xi_v_history[i, :] = self.xi_v
        self.xi_x_history[i, :] = self.xi_x
        self.xi_eta_history[i, :] = self.xi_eta
        self.xi_theta_history[i, :] = self.xi_theta
        self.xi_gamma_history[i, :] = self.xi_gamma


def free_energy(self, i):
    self.prediction_errors(i)
    
    return .5 * (self.eps_v.t() @ self.xi_v + self.eps_x.t() @ self.xi_x + self.eps_eta.t() @ self.xi_eta + \
                self.eps_theta.t() @ self.xi_theta + self.eps_gamma.t() @ self.xi_gamma + \
                ((self.n + self.r + self.p + self.h) * torch.log(2*torch.tensor([[math.pi]])) - torch.logdet(self.Pi_z) - torch.logdet(self.Pi_w) - \
                torch.logdet(self.Pi_theta) - torch.logdet(self.Pi_gamma)).sum())           # TODO: add more terms due to the variational Gaussian approximation?