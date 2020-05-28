# This module containes the definition of a single layer of a generic probabilistc model

import numpy as np
import torch
import functions
from scipy.linalg import sqrtm

_large = np.exp(32.)

# class odeSolver():
#     def __init__(self):
        # TODO: write the code for the local linearisation method (Ozaki)
        # TODO: write the code for some simpler ODE-SDE solver https://github.com/rtqichen/torchdiffeq
        # https://pypi.org/project/DESolver/ http://docs.pyro.ai/en/stable/contrib.tracking.html#module-pyro.contrib.tracking.extended_kalman_filter


class layer():
    def __init__(self, T, dt, A, F, B=0, G=0, Sigma_w=[], Sigma_z=[], Sigma_v=[], D=0, E=0, p=0, h=0, e_n=0, e_r=0, e_p=0, e_h=0, phi=2., history=0):
        self.T = T
        self.dt = dt
        self.iterations = int(T/dt)
        self.history = history
        
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
        self.x = functions.kronecker(torch.eye(self.e_n+1), torch.rand(self.n, 1, requires_grad = True))             # states
        self.v = functions.kronecker(torch.eye(self.e_r+1), torch.zeros(self.r, 1, requires_grad = True))             # inputs
        self.eta_v = functions.kronecker(torch.eye(self.e_r+1), torch.zeros(self.r, 1, requires_grad = True))         # prior on inputs
        
        ## parameters and hyperparameters ##
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
        
        self.Pi_theta = (self.D @ self.D.t()).pinverse()
        self.Pi_gamma = (self.E @ self.E.t()).pinverse()


        ## noise ##
        # TODO: include multiple different correlations if necessary
        self.phi = phi                                                                  # smoothness of temporal correlations

        if len(Sigma_z) > 0:                                                            # if the model includes system noise
            self.H = torch.from_numpy(sqrtm(Sigma_z))
            self.z = H @ functions.spm_DEM_z(self.m, self.phi, self.T, self.dt)
            # self.S_inv_z = functions.temporalPrecisionMatrix(self.e_n+1, self.phi)      # temporal precision matrix observation noise, roughness
            # self.Sigma_z = functions.kronecker(self.S_inv_z, Sigma_z)                   # precision matrix observation noise including higher embedding orders # TODO: find a torch based version of the krocker product
            # self.Pi_z = self.Sigma_z.pinverse()                                         # covariance matrix observation noise including higher embedding orders
            # self.H = torch.sqrt(self.Sigma_z)                                           # observation noise matrix, # FIXME: assuming independent noise across dimensions
            # self.z = functions.kronecker(torch.eye(self.e_n+1), torch.randn(self.m, 1)) # observation noise, as derivative of Wiener process
        else:
            # self.Sigma_z = torch.tensor([_large])
            # self.Pi_z = self.Sigma_z.pinverse()
            self.H = torch.zeros(1)
            self.z = torch.zeros(1)

        if len(Sigma_w) > 0:                                                            # if the model includes measurement noise
            self.C = torch.from_numpy(sqrtm(Sigma_w))
            self.w = H @ functions.spm_DEM_z(self.n, self.phi, self.T, self.dt)
            # self.S_inv_w = functions.temporalPrecisionMatrix(self.e_n+1, self.phi)      # temporal precision matrix system noise, roughness
            # self.Sigma_w = functions.kronecker(self.S_inv_w, Sigma_w)                   # precision matrix system noise including higher embedding orders
            # self.Pi_w = self.Sigma_w.pinverse()                                         # covariance matrix system noise including higher embedding orders
            # self.C = torch.sqrt(self.Sigma_w)                                           # system noise matrix, # FIXME: assuming independent noise across dimensions, we need something like sqrtm for matlab
            # self.w = functions.kronecker(torch.eye(self.e_n+1), torch.randn(self.n, 1)) # system noise, as derivative of Wiener process
        else:
            # self.Sigma_w = torch.tensor([_large])
            # self.Pi_w = self.Sigma_w.pinverse()
            self.C = torch.zeros(1)
            self.w = torch.zeros(1)
        
        if len(Sigma_v) > 0:                                                            # if the model includes uncertainty on inputs
            self.S_v = functions.smoothnessMatrix(self.e_r+1, self.phi)                 # temporal precision matrix system noise
            self.Pi_v = torch.from_numpy(np.kron(self.S_v, Sigma_v))                    # precision matrix system noise including higher embedding orders
            self.Sigma_v = self.Pi_v.pinverse()                                         # covariance matrix system noise including higher embedding orders
        else:
            self.Sigma_v = torch.tensor([[_large]])
            self.Pi_v = self.Sigma_v.pinverse()

        ## history ##
        # TODO: this should be used with parsimony to avoid the RAM blowing up for big models
        self.y_history = torch.zeros(self.history, *self.y.shape)
        self.x_history = torch.zeros(self.history, *self.x.shape)
        self.v_history = torch.zeros(self.history, *self.v.shape)
        self.eta_v_history = torch.zeros(self.history, *self.eta_v.shape)

        self.w_history = torch.zeros(self.history, *self.w.shape)
        self.z_history = torch.zeros(self.history, *self.z.shape)
    
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

    def free_energy(self): 
        return functions.F(self.A, self.F, self.B, self.C, self.G, self.H, self.D, self.E, self.n, self.r, self.p, self.h, self.e_n, self.e_r, self.e_p, self.e_h, self.y, self.x, self.v, self.eta_v, self.theta, self.eta_theta, self.gamma, self.eta_gamma, self.Pi_z, self.Pi_w, self.Pi_v, self.Pi_theta, self.Pi_gamma)

    def step(self, dt):
        # FIXME: Choose and properly implement a numerical solver (or multiple ones?) Euler-Maruyama, Local linearisation, Milner, etc.
        self.dw = functions.Diff(self.w, self.n, self.e_n+1)
        self.dz = functions.Diff(self.z, self.m, self.e_n+1)
        self.dx = self.f() + self.C @ self.w

        self.w += dt * self.dw
        self.z += dt * self.dz
        self.x += dt * self.dx

        print(self.w, self.dw)

        self.y = self.g() + self.H @ self.z

        # for i in range(1, self.n):
        #     self.y[i*self.n:i*self.n+self.n, i*self.n:i*self.n+self.n]


        ### from spm_ADEM_diff

        # u.v{1}  = spm_vec(vi);
        # u.x{2}  = spm_vec(fi) + u.w{1};
        # for i = 2:(n - 1)
        #     u.v{i}     = dg.dv*u.v{i} + dg.dx*u.x{i} + dg.da*u.a{i} + u.z{i};
        #     u.x{i + 1} = df.dv*u.v{i} + df.dx*u.x{i} + df.da*u.a{i} + u.w{i};
        # end
        # u.v{n}  = dg.dv*u.v{n} + dg.dx*u.x{n} + dg.da*u.a{n} + u.z{n};

    # TODO: check how to implement numerical methods for differential equations with pytorch
    # (remember that we need a dt and, consequently, a way to handle non-discrete steps)
    # TODO: define block matrix (multivariate case in arbitrary embedding orders) 
    # for updates of SDEs similar to what can be found in 'Generalised filtering'
    # def forward(self, method=0):
    #     return
        # methods:
        # 0: Local linearisation
        # 1: Euler-Maruyama
    
    def setObservations(self, y):
        self.y = y

    def save_history(self, i):
        self.y_history[i, :, :] = self.y
        self.x_history[i, :, :] = self.x
        self.v_history[i, :, :] = self.v
        self.eta_v_history[i, :, :] = self.eta_v

        self.w_history[i, :, :] = self.w
        self.z_history[i, :, :] = self.z
