# This module containes the definition of a generic probabilistc model

import numpy as np
import torch
from Functions import *

# class odeSolver():
#     def __init__(self):
        # TODO: write the code for the local linearisation method (Ozaki)
        # TODO: write the code for some simpler ODE-SDE solver https://github.com/rtqichen/torchdiffeq
        # https://pypi.org/project/DESolver/ http://docs.pyro.ai/en/stable/contrib.tracking.html#module-pyro.contrib.tracking.extended_kalman_filter

class layer():
    def __init__(self, A, F, B=0, C=[], G=[], H=[], m=0, n=0, r=0, p=0, h=0, e_m=0, e_n=0, e_r=0, e_p=0, e_h=0, phi = 2.):
        
        self.e_m = e_m                                                          # embedding dimension observations
        self.e_n = e_n                                                          # embedding dimension hidden states
        self.e_r = e_r                                                          # embedding dimension inputs
        self.e_p = e_p                                                          # embedding dimension parameters
        self.e_h = e_h                                                          # embedding dimension hyperparameters

        self.m = len(F)                                                         # observations dimension
        self.n = len(A)                                                         # hidden states dimension
        if B == 0:
            self.r = len(B)                                                     # inputs dimension (dynamics)
            self.B = torch.zeros(1)                                             # input matrix (dynamics), default
        else:
            self.r = len(B[0])                                                  # inputs dimension (dynamics)
            
        if len(G) == 0:                                                         
            self.G = torch.zeros(1)                                             # input matrix (observations), default
        self.p = p                                                              # parameters dimension
        self.h = h                                                              # hyperparameters dimension

        self.A = torch.from_numpy(np.kron(np.eye(self.e_n+1), A))               # state transition matrix
        self.B = torch.from_numpy(np.kron(np.eye(self.e_r+1), B))               # input matrix (dynamics)
        self.F = torch.from_numpy(np.kron(np.eye(self.e_m+1), F))               # observation matrix
        self.G = torch.from_numpy(np.kron(np.eye(self.e_r+1), G))               # input matrix (observations)

        # TODO: for nonlinear systems, higher embedding orders of y, x, v 
        # should contain derivatives of functions f and g
        self.y = torch.from_numpy(np.kron(np.eye(self.e_m+1), torch.zeros(self.m, 1)))            # observations
        self.x = torch.from_numpy(np.kron(np.eye(self.e_n+1), torch.zeros(self.n, 1)))            # states
        self.v = torch.from_numpy(np.kron(np.eye(self.e_r+1), torch.zeros(self.r, 1)))            # inputs

        self.theta = torch.zeros(self.p, self.e_p+1)                            # parameters
        self.gamma = torch.zeros(self.h, self.e_h+1)                            # hyperparameters


        # noise
        # TODO: include multiple different correlations if necessary
        self.phi = phi                                                          # smoothness of temporal correlations

        if len(H) > 0:                                                              # if the model includes noise
            self.S_z = smoothnessMatrix(self.e_m, self.phi)                         # temporal precision matrix observation noise
            self.Pi_z = np.kron(self.S_z, np.linalg.pinv(np.dot(H, H.transpose()))) # precision matrix observation noise including higher embedding orders
            self.Sigma_z = np.linalg.pinv(self.Pi_z)                                # covariance matrix observation noise including higher embedding orders
            self.H = numpy.sqrt(Sigma_z)                                            # observation noise matrix, # FIXME: assuming independent noise across dimensions
            self.z = np.kron(np.eye(elf.e_m+1), np.random.randn(m, 1))              # observation noise, as derivative of Wiener process
        else:
            self.H = torch.zeros(1)
            self.z = torch.zeros(1)

        if len(C) > 0:                                                              # if the model includes noise
            self.S_w = smoothnessMatrix(self.e_n, self.phi)                         # temporal precision matrix system noise
            self.Pi_w = np.kron(self.S_w, np.linalg.pinv(np.dot(C, C.transpose()))) # precision matrix system noise including higher embedding orders
            self.Sigma_w = np.linalg.pinv(self.Pi_w)                                # covariance matrix system noise including higher embedding orders
            self.C = numpy.sqrt(Sigma_w)                                            # system noise matrix, # FIXME: assuming independent noise across dimensions
            self.w = np.kron(np.eye(elf.e_n+1), np.random.randn(n, 1))              # system noise, as derivative of Wiener process
        else:
            self.C = torch.zeros(1)
            self.w = torch.zeros(1)
    
    def f(self):
        # TODO: generalise this to include nonlinear treatments
        return self.A @ self.x + self.B @ self.v + self.C @ self.w
        
    def g(self, func):
        return 1

    def h(x, v, theta):
        return 1

    def p_1(x, v, theta):
        return 1

    def p_2(x, v, theta):
        return 1
    # TODO: check how to implement numerical methods for differential equations with pytorch
    # (remember that we need a dt and, consequently, a way to handle non-discrete steps)
    # TODO: define block matrix (multivariate case in arbitrary embedding orders) 
    # for updates of SDEs similar to what can be found in 'Generalised filtering'
    def forward(self, method=0):
        return
        # methods:
        # 0: Local linearisation
        # 1: Euler-Maruyama
        

# class HDM():
#     def __init__(self, l, m=1, n=0, r=0, p=0, h=0, e_m=0, e_n=0, e_r=0, e_p=0, e_h=0):
#         self.l = l                               # number of layers

#         self.m = m                               # observations dimension
#         self.n = n                               # hidden states dimension
#         self.r = r                               # inputs dimension
#         self.p = p                               # parameters dimension
#         self.h = h                               # hyperparameters dimension

#         self.e_m = e_m                           # embedding dimension observations
#         self.e_n = e_n                           # embedding dimension hidden states
#         self.e_r = e_r                           # embedding dimension inputs
#         self.e_p = e_p                           # embedding dimension parameters
#         self.e_h = e_h                           # embedding dimension hyperparameters

#         for i in range(self.l):
#             self.layers[i] = layer()            # create layers

#     def addLayer(self, llayer, level):
#         self.layers[level] = llayer

#     # FIXME: check for boundary conditions (first and last layer)
#     # FIXME: make sure that 'self.layers' is an array of objects
#     # FIXME: check the arguments for 'forward' in 'layers' class
#     def forward(self):
#         for i in range(self.l):
#             output = self.layers[i](output)
#         return output

#     # FIXME: errors per layer, add and use layer argument
#     def epsilon_z(self, layer):
#         return self.y[layer,:,:] - self.g(self)
    
#     # TODO: check how weighted prediction errors are defined
#     def xi_z(self, layer):
#         return torch.bmm(Pi_z, epsilon_z(self))
