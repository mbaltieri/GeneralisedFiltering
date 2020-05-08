# This module will contain general functions used for the inversion of generative models

import torch
import autograd.numpy as np                 # autograd now implemented in JAX, can be replaced
from autograd import grad                   # TODO: move this to pure pytorch

torch.set_default_dtype(torch.float64)

### FUNCTIONS ###

def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

def sigmoid(x):
#    return x
    return np.tanh(x)

def dsigmoid(x):
#    return x
    return 1 - np.tanh(x)**2

# autocorrelation function (Gaussian)
def rho(h, phi):
    return np.exp(-.5*h**2/phi)

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
def smoothnessMatrix(embedding_orders, phi):
    h = 0.0                                                             # lag
    derivative_order = (embedding_orders-1)*2+1
    rho_tilde = np.zeros(derivative_order,)                             # array of autocorrelations
    S_inv = np.zeros((embedding_orders,embedding_orders))               # temporal roughness
    S = np.zeros((embedding_orders,embedding_orders))                   # temporal smoothness

    drho = findDerivatives(derivative_order)
    rho_tilde[0] = rho(h, phi)
    for i in range(1, derivative_order):
        # rho_tilde[i] = dnrho(h, phi, i-1)
        rho_tilde[i] = drho[i-1](h, phi)

    for i in range(embedding_orders):
        for j in range(embedding_orders):
            S_inv[i, j] = np.power(-1, (i*j))*rho_tilde[i+j]
    S = np.linalg.inv(S_inv)

    return torch.from_numpy(S)


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
    

def FreeEnergy(y, mu_x, mu_v, mu_pi_z, mu_pi_w, A_gm, B_gm, F_gm):
    return .5 * (np.sum(np.dot(np.dot((y - np.dot(F_gm, mu_x)).transpose(), mu_pi_z), (y - np.dot(F_gm, mu_x)))) + \
                np.sum(np.dot(np.dot((mu_x[:, 1:] - f_gm(mu_x[:, :-1], mu_v[:, :-1], A_gm, B_gm)).transpose(), mu_pi_w), (mu_x[:, 1:] - f_gm(mu_x[:, :-1], mu_v[:, :-1], A_gm, B_gm)))) - \
                np.trace(np.log(mu_pi_z * mu_pi_w)))

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

    return .5 * (eps_v.t() @ Pi_z @ eps_v + eps_x.t() @ Pi_w @ eps_x + eps_eta.t() @ Pi_v @ eps_eta + \
                 eps_theta.t() @ Pi_theta @ eps_theta + eps_gamma.t() @ Pi_gamma @ eps_gamma + \
                 (n + r + p + h) * np.log(2*np.pi) - np.log(Pi_z.det()) - np.log(Pi_w.det()) - \
                 np.log(Pi_theta.det()) - np.log(Pi_gamma.det())).sum()     # TODO: add more terms due to the variational Gaussian approximation?