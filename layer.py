# This module containes the definition of a single layer of a generic probabilistc model

import math

# import numpy as np
import torch
import functions
# from scipy.linalg import sqrtm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_large = torch.exp(torch.tensor(32.))
_small = torch.exp(torch.tensor(-32.))

# class odeSolver():
#     def __init__(self):
        # TODO: write the code for the local linearisation method (Ozaki)
        # TODO: write the code for some simpler ODE-SDE solver https://github.com/rtqichen/torchdiffeq
        # https://pypi.org/project/DESolver/ http://docs.pyro.ai/en/stable/contrib.tracking.html#module-pyro.contrib.tracking.extended_kalman_filter


class layer():
    def __init__(self, T, dt, A, F, B_v=[], B_a=[], G=[], Sigma_w=[], Sigma_z=[], Sigma_v=[], D=0, E=0, p=0, h=0, e_n=0, e_r=0, e_p=0, e_h=0, phi=2., dyda=[], eta_v=0):
        self.T = T
        self.dt = dt
        self.iterations = int(T/dt)
        
        self.e_n = e_n                                                          # embedding dimension hidden states
        self.e_r = e_r                                                          # embedding dimension inputs
        self.e_p = e_p                                                          # embedding dimension parameters
        self.e_h = e_h                                                          # embedding dimension hyperparameters

        self.e_sim = max(self.e_n, self.e_r)                                    # embedding dimension for simulations, i.e., the maximum one    #TODO: big zero matrices? If so, check how to define sparse ones

        self.m = len(F)                                                         # observations dimension
        self.n = len(A)                                                         # hidden states dimension
        if (len(B_v) != 0) or (len(B_a) != 0):
            self.r = max(len(B_v), len(B_a), self.n)                            # for consistency, create matrices of the same size
            if len(B_v) == 0:
                B_v = torch.zeros(self.r, self.r, device=DEVICE)                # inputs dimension (external dynamics)
            else:
                B_a = torch.zeros(self.r, self.r, device=DEVICE)                # input matrix (self-generated dynamics), default
        else:
            self.r = 1                                                          # inputs dimension (external dynamics)
            B_v = torch.zeros(self.r, self.r, device=DEVICE)                              # input matrix (external dynamics), default
            B_a = torch.zeros(self.r, self.r, device=DEVICE)                              # input matrix (self-generated dynamics), default

        self.sim = max(self.n, self.r)                                          # dimension for simulations, i.e., the maximum one              #TODO: big zero matrices? If so, check how to define sparse ones


        # if len(B_v) == 0:
        #     self.r = 1                                                          # inputs dimension (external dynamics)
        #     B_v = torch.zeros(1, 1, device=DEVICE)                              # input matrix (external dynamics), default
        # else:
        #     self.r = len(B_v[0])                                                # inputs dimension (external dynamics)
        # if len(B_a) == 0:
        #     B_a = torch.zeros(1, 1, device=DEVICE)                              # input matrix (self-generated dynamics), default
        # else:
        #     self.r = len(B_a[0])                                                # inputs dimension (self-generated dynamics)



        if len(dyda) == 0:
            dyda = torch.zeros(self.sim, self.r, device=DEVICE)                             # input matrix (self-generated dynamics), default
        if len(G) == 0:
            G = torch.zeros(self.r, self.r, device=DEVICE)                                # input matrix (observations), default

        self.p = p                                                              # parameters dimension
        self.h = h                                                              # hyperparameters dimension

        if D == 0:                                                              # if there are no parameters # TODO: in general or to learn?
            self.D = torch.tensor([[_small]], device=DEVICE)
        if E == 0:                                                              # if there are no hyperparameters # TODO: in general or to learn?
            self.E = torch.tensor([[_small]], device=DEVICE)


        # add padding
        if self.n > self.r:                                                     # for now, assuming always self.n >= self.r, if not, the noise needs to be adapted
            B_v = torch.nn.functional.pad(B_v, (0, 0, 0, self.n - self.r))
            B_v = torch.nn.functional.pad(B_v, (0, self.n - self.r, 0, 0))

            B_a = torch.nn.functional.pad(B_a, (0, 0, 0, self.n - self.r))
            B_a = torch.nn.functional.pad(B_a, (0, self.n - self.r, 0, 0))

            G = torch.nn.functional.pad(G, (0, 0, 0, self.n - self.r))
            G = torch.nn.functional.pad(G, (0, self.n - self.r, 0, 0))
        elif self.n < self.r:
            print('Not implemented, quitting.')
            quit()
        
        # if self.e_n > self.e_r:                                                 # for now, assuming always self.e_n >= self.e_r, if not, the noise needs to be adapted
        #     B_v = torch.nn.functional.pad(B_v, (0, self.e_n - self.e_r, 0, 0))
        #     B_a = torch.nn.functional.pad(B_a, (0, self.e_n - self.e_r, 0, 0))


        # create block matrices considering higher embedding orders
        self.A = functions.kronecker(torch.eye(self.e_sim+1), A)                  # state transition matrix
        self.B_v = functions.kronecker(torch.eye(self.e_sim+1), B_v)              # input matrix (external dynamics)
        self.B_a = functions.kronecker(torch.eye(self.e_sim+1), B_a)              # input matrix (self-generated dynamics)
        self.F = functions.kronecker(torch.eye(self.e_sim+1), F)                  # observation matrix
        self.G = functions.kronecker(torch.eye(self.e_sim+1), G)                  # input matrix (observations)
        self.dyda = functions.kronecker(torch.ones(self.e_sim+1,1), dyda)            # (direct) influence of actions on observations  

        # TODO: for nonlinear systems, higher embedding orders of y, x, v 
        # should contain derivatives of functions f and g
        # self.y = functions.kronecker(torch.eye(self.e_sim+1), torch.zeros(self.m, 1, requires_grad = True))             # observations
        # self.x = functions.kronecker(torch.eye(self.e_sim+1), torch.rand(self.n, 1, requires_grad = True))              # states
        # self.v = functions.kronecker(torch.eye(self.e_sim+1), torch.zeros(self.r, 1, requires_grad = True))             # inputs
        # self.eta_v = functions.kronecker(torch.eye(self.e_sim+1), torch.zeros(self.r, 1, requires_grad = True))         # prior on inputs


        self.y = torch.zeros(self.sim*(self.e_sim+1), 1, requires_grad = True, device = DEVICE)             # observations
        self.x = torch.zeros(self.sim*(self.e_sim+1), 1, requires_grad = True, device = DEVICE)             # states
        self.v = torch.zeros(self.sim*(self.e_sim+1), 1, requires_grad = True, device = DEVICE)             # inputs (GM) / external forces (GP)
        self.a = torch.zeros(self.sim*(self.e_sim+1), 1, requires_grad = True, device = DEVICE)             # self-produced actions (GP)
        # self.d = torch.zeros(self.r*(self.e_sim+1), 1, requires_grad = True, device = DEVICE)             # external forces (GP)
        self.eta_v = torch.zeros(self.sim*(self.e_sim+1), 1, requires_grad = True, device = DEVICE)         # prior on inputs
        
        if eta_v != 0:                                                                                  # if there is a prior, assign it
            try:
                self.eta_v = functions.kronecker(torch.eye(self.e_sim+1), eta_v)
            except RuntimeError:
                print("Prior dimensions don't match. Please check and try again.")
                quit()
        
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
        self.invphi = 1/phi                                                             # roughness of temporal correlations

        if len(Sigma_z) > 0:                                                            # if the model includes system noise
            # self.H = functions.symsqrt(Sigma_z)
            self.zSmoothened = functions.spm_DEM_z(self.m, self.phi, self.T, self.dt)
            self.S_inv_z = functions.spm_DEM_R(self.e_sim+1, self.invphi)                 # temporal precision matrix observation noise, roughness
            self.Sigma_z = functions.kronecker(self.S_inv_z, Sigma_z)                   # covariance matrix observation noise including higher embedding orders # TODO: find a torch based version of the krocker product
            self.Pi_z = self.Sigma_z.pinverse()                                         # precision matrix observation noise including higher embedding orders
            self.H = functions.symsqrt(self.Sigma_z)
            # self.H = torch.sqrt(self.Sigma_z)                                           # observation noise matrix, # FIXME: assuming independent noise across dimensions
            # self.z = functions.kronecker(torch.eye(self.e_sim+1), torch.randn(self.m, 1)) # observation noise, as derivative of Wiener process
        else:
            # self.Sigma_z = torch.tensor([_large])
            # self.Pi_z = self.Sigma_z.pinverse()
            self.H = torch.zeros(1)
            self.z = torch.zeros(1)

        if len(Sigma_w) > 0:                                                            # if the model includes measurement noise
            # self.C = functions.symsqrt(Sigma_w)
            self.wSmoothened = functions.spm_DEM_z(self.n, self.phi, self.T, self.dt)
            self.S_inv_w = functions.spm_DEM_R(self.e_sim+1, self.invphi)                 # temporal precision matrix system noise, roughness
            self.Sigma_w = functions.kronecker(self.S_inv_w, Sigma_w)                   # covariance matrix system noise including higher embedding orders
            self.Pi_w = self.Sigma_w.pinverse()                                         # precision matrix system noise including higher embedding orders
            self.C = functions.symsqrt(self.Sigma_w)
            # self.C = torch.sqrt(self.Sigma_w)                                           # system noise matrix, # FIXME: assuming independent noise across dimensions, we need something like sqrtm for matlab
            # self.w = functions.kronecker(torch.eye(self.e_sim+1), torch.randn(self.n, 1)) # system noise, as derivative of Wiener process
        else:
            # self.Sigma_w = torch.tensor([_large])
            # self.Pi_w = self.Sigma_w.pinverse()
            self.C = torch.zeros(1)
            self.w = torch.zeros(1)
        
        if len(Sigma_v) > 0:                                                            # if the model includes uncertainty on inputs
            self.S_v = functions.smoothnessMatrix(self.e_sim+1, self.phi)                 # temporal precision matrix system noise
            self.Pi_v = functions.kronecker(self.S_v, Sigma_v)                          # precision matrix system noise including higher embedding orders
            self.Sigma_v = self.Pi_v.pinverse()                                         # covariance matrix system noise including higher embedding orders
        else:
            self.Sigma_v = _large * torch.ones(self.sim*(self.e_sim+1), self.sim*(self.e_sim+1))
            self.Pi_v = self.Sigma_v.pinverse()

        ## history ##
        # TODO: this should be used with parsimony to avoid the RAM blowing up for big models
        self.y_history = torch.zeros(self.iterations, *self.y.shape)
        self.x_history = torch.zeros(self.iterations, *self.x.shape)
        self.v_history = torch.zeros(self.iterations, *self.v.shape)
        self.a_history = torch.zeros(self.iterations, *self.a.shape)
        self.eta_v_history = torch.zeros(self.iterations, *self.eta_v.shape)

        self.w_history = torch.zeros(self.iterations, *self.x.shape)
        self.z_history = torch.zeros(self.iterations, *self.y.shape)

        self.F_history = torch.zeros(self.iterations, 1)


        ### Run some simple checks
        if self.A.size(0) != self.C.size(0):
            print('State transition matrix and process noise matrix sizes don\'t match. Please check and try again.')
            quit()
        if self.F.size(0) != self.H.size(0):
            print('Measurement matrix and measurement noise matrix sizes don\'t match. Please check and try again.')
            quit()

    
    def f(self, i):
        return functions.f(self.A, self.B_v, self.B_a, self.x, self.v, self.a)

    def g(self, i):
        return functions.g(self.F, self.G, self.x, self.v)

    def k_theta(self):                                                             # TODO: for simplicity we hereby assume that parameters are independent of other variables
        return functions.k_theta(self.D, self.eta_theta)
    
    def k_gamma(self):                                                             # TODO: for simplicity we hereby assume that hyperparameters are independent of other variables
        return functions.k_theta(self.E, self.eta_gamma)

    def prediction_errors(self, i):
        self.eps_v = self.y - self.g(i)
        self.eps_x = functions.Diff(self.x, self.n, self.e_sim+1) - self.f(i)
        self.eps_eta = self.v - self.eta_v
        self.eps_theta = self.theta - self.k_theta()
        self.eps_gamma = self.gamma - self.k_gamma()

    def free_energy(self, i): 
        # self.F_history[i] = functions.F(self.A, self.F, self.B, self.C, self.G, self.H, self.D, self.E, self.n, self.r, self.p, self.h, self.e_sim, self.e_sim, self.e_p, self.e_h, self.y, self.x, self.v, self.eta_v, self.theta, self.eta_theta, self.gamma, self.eta_gamma, self.Pi_z, self.Pi_w, self.Pi_v, self.Pi_theta, self.Pi_gamma)
        self.prediction_errors(i)

        self.F_history[i] = .5 * (self.eps_v.t() @ self.Pi_z @ self.eps_v + self.eps_x.t() @ self.Pi_w @ self.eps_x + self.eps_eta.t() @ self.Pi_v @ self.eps_eta + \
                            self.eps_theta.t() @ self.Pi_theta @ self.eps_theta + self.eps_gamma.t() @ self.Pi_gamma @ self.eps_gamma + \
                            ((self.n + self.r + self.p + self.h) * torch.log(2*torch.tensor([[math.pi]])) - torch.logdet(self.Pi_z) - torch.logdet(self.Pi_w) - \
                            torch.logdet(self.Pi_theta) - torch.logdet(self.Pi_gamma)).sum())     # TODO: add more terms due to the variational Gaussian approximation?

        return .5 * (self.eps_v.t() @ self.Pi_z @ self.eps_v + self.eps_x.t() @ self.Pi_w @ self.eps_x + self.eps_eta.t() @ self.Pi_v @ self.eps_eta + \
                    self.eps_theta.t() @ self.Pi_theta @ self.eps_theta + self.eps_gamma.t() @ self.Pi_gamma @ self.eps_gamma + \
                    ((self.n + self.r + self.p + self.h) * torch.log(2*torch.tensor([[math.pi]])) - torch.logdet(self.Pi_z) - torch.logdet(self.Pi_w) - \
                    torch.logdet(self.Pi_theta) - torch.logdet(self.Pi_gamma)).sum())

    def step(self, i):
        # FIXME: Choose and properly implement a numerical solver (or multiple ones?) Euler-Maruyama, Local linearisation, Milner, etc.

        # TODO: The following code works (?) for linear functions (and not nonlinear ones) in virtue of the fact that generalised coordinates for linear models are trivial; for nonlinear models, see snippet "from spm_ADEM_diff"
        # self.dw = functions.Diff(self.w, self.n, self.e_sim+1)
        # self.dz = functions.Diff(self.z, self.m, self.e_sim+1)

        self.w = functions.spm_DEM_embed(self.wSmoothened, self.e_sim+1, i, dt=1.)                # FIXME: This I don't fully understand, but if we impose dt < 1. here we get a weird behaviour, e.g., dt = 0.1 only the first 1/10 of the sequence is considered and then the noise is flat
        self.z = functions.spm_DEM_embed(self.zSmoothened, self.e_sim+1, i, dt=1.)                # FIXME: After chacking the above, find out if the precisions needs to be changed, following equation 55 of the DEM paper. So far no hint in the code, but maybe in spm_DEM_R?

        # self.dx = self.f(i) + self.C @ self.w[:,i,None]
        self.dx = self.f(i) + self.C @ self.w

        # self.w[i+1,:,:] = self.w[i,:,:] + self.dt * self.dw
        # self.z[i+1,:,:] = self.z[i,:,:] + self.dt * self.dz
        self.x = self.x + self.dt * self.dx

        # self.y[i+1,:,:] = self.g(i) + self.H @ self.z[:,i,None]
        self.y = self.g(i) + self.H @ self.z

        self.save_history(i)


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
        self.y = y.detach()
    
    def save_history(self, i):
        self.y_history[i, :] = self.y
        self.x_history[i, :] = self.x
        self.v_history[i, :] = self.v
        self.a_history[i, :] = self.a
        self.eta_v_history[i, :] = self.eta_v

        if hasattr(self, 'w'):
            self.w_history[i, :] = self.w
            self.z_history[i, :] = self.z
