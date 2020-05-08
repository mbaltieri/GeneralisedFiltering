# This module containes the definition of a generic probabilistc model

import numpy as np
# from jax import grad
import torch
import functions

_large = np.exp(32.)

# class odeSolver():
#     def __init__(self):
        # TODO: write the code for the local linearisation method (Ozaki)
        # TODO: write the code for some simpler ODE-SDE solver https://github.com/rtqichen/torchdiffeq
        # https://pypi.org/project/DESolver/ http://docs.pyro.ai/en/stable/contrib.tracking.html#module-pyro.contrib.tracking.extended_kalman_filter


class layer():
    def __init__(self, A, F, B=0, G=0, Sigma_w=[], Sigma_z=[], Sigma_v=[], D=0, E=0, p=0, h=0, e_n=0, e_r=0, e_p=0, e_h=0, phi=2., dt=0.01):
        self.dt = dt
        
        self.e_n = e_n                                                          # embedding dimension hidden states
        self.e_r = e_r                                                          # embedding dimension inputs
        self.e_p = e_p                                                          # embedding dimension parameters
        self.e_h = e_h                                                          # embedding dimension hyperparameters

        self.m = len(F)                                                         # observations dimension
        self.n = len(A)                                                         # hidden states dimension
        if B == 0:
            self.r = 1                                                          # inputs dimension (dynamics)
            B = torch.zeros(1, 1)                                               # input matrix (dynamics), default
        else:
            self.r = len(B[0])                                                  # inputs dimension (dynamics)
        if G == 0:
            G = torch.zeros(1, 1)                                               # input matrix (observations), default
        self.p = p                                                              # parameters dimension
        self.h = h                                                              # hyperparameters dimension

        if D == 0:                                                              # if there are no parameters # TODO: in general or to learn?
            self.D = torch.tensor([[_large]])
        if E == 0:                                                              # if there are no hyperparameters # TODO: in general or to learn?
            self.E = torch.tensor([[_large]])

        # create block matrices considering higher embedding orders
        self.A = functions.kronecker(torch.eye(self.e_n+1), A)                  # state transition matrix
        self.B = functions.kronecker(torch.eye(self.e_r+1), B)                  # input matrix (dynamics)
        self.F = functions.kronecker(torch.eye(self.e_n+1), F)                  # observation matrix
        self.G = functions.kronecker(torch.eye(self.e_r+1), G)                  # input matrix (observations)

        # TODO: for nonlinear systems, higher embedding orders of y, x, v 
        # should contain derivatives of functions f and g
        self.y = functions.kronecker(torch.eye(self.e_n+1), torch.zeros(self.m, 1, requires_grad = True))             # observations
        self.x = functions.kronecker(torch.eye(self.e_n+1), torch.zeros(self.n, 1, requires_grad = True))             # states
        self.v = functions.kronecker(torch.eye(self.e_r+1), torch.zeros(self.r, 1, requires_grad = True))             # inputs
        self.eta_v = functions.kronecker(torch.eye(self.e_r+1), torch.zeros(self.r, 1, requires_grad = True))         # prior on inputs

        if p == 0:                                                                  # if there are no parameters # TODO: in general or to learn?
            self.theta = torch.zeros(1)
            self.eta_theta = torch.zeros(1)
        else:
            self.theta = torch.zeros(self.p, self.e_p+1)                            # parameters
            self.eta_theta = torch.zeros(self.p, self.e_p+1)                        # prior on parameters

        if h == 0:                                                                  # if there are no hyperparameters # TODO: in general or to learn?
            self.gamma = torch.zeros(1)
            self.eta_gamma = torch.zeros(1)
        else:
            self.gamma = torch.zeros(self.h, self.e_h+1)                            # hyperparameters
            self.eta_gamma = torch.zeros(self.h, self.e_h+1)                        # prior on hyperparameters


        # noise
        # TODO: include multiple different correlations if necessary
        self.phi = phi                                                                  # smoothness of temporal correlations

        if len(Sigma_w) > 0:                                                            # if the model includes system noise
            self.S_z = functions.smoothnessMatrix(self.e_n+1, self.phi)                 # temporal precision matrix observation noise
            self.Pi_z = torch.from_numpy(np.kron(self.S_z, Sigma_z))                    # precision matrix observation noise including higher embedding orders # TODO: find a torch based version of the krocker product
            self.Sigma_z = self.Pi_z.pinverse()                                         # covariance matrix observation noise including higher embedding orders
            self.H = torch.sqrt(self.Sigma_z)                                           # observation noise matrix, # FIXME: assuming independent noise across dimensions
            self.z = functions.kronecker(torch.eye(self.e_n+1), torch.randn(self.m, 1)) # observation noise, as derivative of Wiener process
        else:
            self.Sigma_z = torch.tensor([_large])
            self.Pi_z = self.Sigma_z.pinverse()
            self.H = torch.zeros(1)
            self.z = torch.zeros(1)

        if len(Sigma_z) > 0:                                                            # if the model includes measurement noise
            self.S_w = functions.smoothnessMatrix(self.e_n+1, self.phi)                 # temporal precision matrix system noise
            self.Pi_w = torch.from_numpy(np.kron(self.S_w, Sigma_w))                    # precision matrix system noise including higher embedding orders
            self.Sigma_w = self.Pi_w.pinverse()                                         # covariance matrix system noise including higher embedding orders
            self.C = torch.sqrt(self.Sigma_w)                                           # system noise matrix, # FIXME: assuming independent noise across dimensions
            self.w = functions.kronecker(torch.eye(self.e_n+1), torch.randn(self.n, 1)) # system noise, as derivative of Wiener process
        else:
            self.Sigma_w = torch.tensor([_large])
            self.Pi_w = self.Sigma_w.pinverse()
            self.C = torch.zeros(1)
            self.w = torch.zeros(1)
        
        if len(Sigma_v) > 0:                                                            # if the model includes uncertainty on inputs
            self.S_v = functions.smoothnessMatrix(self.e_r+1, self.phi)                 # temporal precision matrix system noise
            self.Pi_v = torch.from_numpy(np.kron(self.S_v, Sigma_v))                    # precision matrix system noise including higher embedding orders
            self.Sigma_v = self.Pi_v.pinverse()                                         # covariance matrix system noise including higher embedding orders
        else:
            self.Sigma_v = torch.tensor([[_large]])
            self.Pi_v = self.Sigma_v.pinverse()
        

        
        self.Pi_theta = (self.D @ self.D.t()).pinverse()
        self.Pi_gamma = (self.E @ self.E.t()).pinverse()
    
    def f(self):
        return functions.f(self.A, self.B, self.x, self.v)

    def g(self):
        return functions.g(self.F, self.G, self.x, self.v)

    def k_theta(self):                                                             # TODO: for simplicity we hereby assume that parameters are independent of other variables
        return functions.k_theta(self.D, self.eta_theta)
    
    def k_gamma(self):                                                             # TODO: for simplicity we hereby assume that hyperparameters are independent of other variables
        return functions.k_theta(self.E, self.eta_gamma)

    def prediction_errors(self):
        self.eps_v = self.y - self.g()
        self.eps_x = functions.Diff(self.x, self.n, self.e_n+1) - self.f()
        self.eps_theta = self.theta - self.k_theta()
        self.eps_gamma = self.gamma - self.k_gamma()
        
    def updateObservations(self, y):
        self.y = y

    def free_energy(self): 
        return functions.F(self.A, self.F, self.B, self.C, self.G, self.H, self.D, self.E, self.n, self.r, self.p, self.h, self.e_n, self.e_r, self.e_p, self.e_h, self.y, self.x, self.v, self.eta_v, self.theta, self.eta_theta, self.gamma, self.eta_gamma, self.Pi_z, self.Pi_w, self.Pi_v, self.Pi_theta, self.Pi_gamma)

    def forward(self, flag=0):
        # One-step Euler-Maruyama
        if flag == 0:                                   # dynamics
            dw = functions.Diff(self.w)
            dz = functions.Diff(self.z)
            dx = f(self) + self.C @ self.w

            self.w += self.dt * dw
            self.z += self.dt * dz
            self.x += self.dt * dx

            self.y = self.g(self) + self.H @ self.z
        else:                                           # inference
            dFdx = self.x.grad

    def backward(self):
        self.F = self.free_energy()
        self.F.backward()
        
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
#     def __init__(self, l, m=1, n=0, r=0, p=0, h=0, e_n=0, e_n=0, e_r=0, e_p=0, e_h=0):
#         self.l = l                               # number of layers

#         self.m = m                               # observations dimension
#         self.n = n                               # hidden states dimension
#         self.r = r                               # inputs dimension
#         self.p = p                               # parameters dimension
#         self.h = h                               # hyperparameters dimension

#         self.e_n = e_n                           # embedding dimension observations
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
