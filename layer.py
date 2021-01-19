# This module containes the definition of a single layer of a generic probabilistc model

import math
import torch
from functions import kronecker, symsqrt, Diff
from smoothNoise import noise
from globalVariables import DEVICE, _small
from integrationSchemes import dx_ll

# class odeSolver():
#     def __init__(self):
        # TODO: write the code for the local linearisation method (Ozaki)
        # TODO: write the code for some simpler ODE-SDE solver https://github.com/rtqichen/torchdiffeq
        # https://pypi.org/project/DESolver/ http://docs.pyro.ai/en/stable/contrib.tracking.html#module-pyro.contrib.tracking.extended_kalman_filter


class layer():
    def __init__(self, flag, T, dt, A, F, B_u=[], B_a=[], G=[], Sigma_w=[], Sigma_z=[], Sigma_v=[], D=0, E=0, p=0, h=0, e_n=0, e_r=0, e_p=0, e_h=0, phi=.5, dyda=[], eta_u=[]):
        self.flag = flag

        self.T = T
        self.dt = dt
        self.iterations = int(T/dt)
        
        self.e_n = e_n                                                          # embedding dimension hidden states
        self.e_r = e_r                                                          # embedding dimension inputs
        self.e_p = e_p                                                          # embedding dimension parameters
        self.e_h = e_h                                                          # embedding dimension hyperparameters

        self.m = len(F)                                                         # observations dimension
        self.n = len(A)                                                       # hidden states dimension (n+1 variables for n equations)

        self.A = A
        self.B_u = B_u
        self.B_a = B_a
        self.F = F
        self.G = G
        self.dyda = dyda

        self.r = max(len(B_u), len(B_a), len(G), 1)                             # number of controls
        self.sim = max(self.n, self.r)                                          # dimension for simulations, i.e., the maximum one              #TODO: big zero matrices? If so, check how to define sparse ones
        self.e_sim = max(self.e_n, self.e_r)                                    # embedding dimension for simulations, i.e., the maximum one    #TODO: big zero matrices? If so, check how to define sparse ones

        ## check control matrices
        self.checkControl()

        self.p = p                                                              # parameters dimension
        self.h = h                                                              # hyperparameters dimension

        if D == 0:                                                              # if there are no parameters # TODO: in general or to learn?
            self.D = torch.tensor([[_small]], device=DEVICE)
        if E == 0:                                                              # if there are no hyperparameters # TODO: in general or to learn?
            self.E = torch.tensor([[_small]], device=DEVICE)


        # create block matrices considering higher embedding orders
        self.A = kronecker(torch.eye(self.e_sim+1), self.A)                          # state transition matrix
        self.B_u = kronecker(torch.eye(self.e_sim+1), self.B_u)                      # input matrix (external dynamics)
        self.B_a = kronecker(torch.eye(self.e_sim+1), self.B_a)                      # input matrix (self-generated dynamics)
        self.F = kronecker(torch.eye(self.e_sim+1), self.F)                          # observation matrix
        self.G = kronecker(torch.eye(self.e_sim+1), self.G)                          # input matrix (observations)
        self.dyda = kronecker(torch.eye(self.e_sim+1), self.dyda)                 # (direct) influence of actions on observations  

        # create variables
        self.y = torch.zeros((self.sim+1) * (self.e_sim+1), 1, requires_grad = True, device = DEVICE)                         # observations
        self.x = torch.normal(0, 10, size=((self.sim+1) * (self.e_sim+1), 1), requires_grad = True, device = DEVICE)          # states
        self.u = torch.zeros((self.sim+1) * (self.e_sim+1), 1, requires_grad = True, device = DEVICE)                         # inputs (GM) / external forces (GP)
        self.a = torch.zeros((self.sim+1) * (self.e_sim+1), 1, requires_grad = True, device = DEVICE)                         # self-produced actions (GP)
        if len(eta_u) == 0:
            self.eta_u = torch.zeros((self.sim+1)*(self.e_sim), 1, requires_grad = True, device = DEVICE)     # prior on inputs
        else:                                                                                               # if there is a prior, assign it
            try:
                # self.eta_u = torch.zeros(self.sim*(self.e_sim+1), 1, requires_grad = True, device = DEVICE)        # FIXME: make sure to update eta_u in step too, otherwise priors won't get propagated
                with torch.no_grad():
                    self.eta_u = kronecker(torch.ones(self.e_sim+1,1), eta_u)
            except RuntimeError:
                print("Prior dimensions don't match. Please check and try again.")
                quit()
        
        ## parameters and hyperparameters ##
        if p == 0:                                                              # if there are no parameters # TODO: in general or to learn?
            self.theta = torch.zeros(1)
            self.eta_theta = torch.zeros(1)
        else:
            self.theta = torch.zeros(self.p, self.e_p+1)                        # parameters
            self.eta_theta = torch.zeros(self.p, self.e_p+1)                    # prior on parameters

        if h == 0:                                                              # if there are no hyperparameters # TODO: in general or to learn?
            self.gamma = torch.zeros(1)
            self.eta_gamma = torch.zeros(1)
        else:
            self.gamma = torch.zeros(self.h, self.e_h+1)                        # hyperparameters
            self.eta_gamma = torch.zeros(self.h, self.e_h+1)                    # prior on hyperparameters
        
        self.Pi_theta = (self.D @ self.D.t()).pinverse()
        self.Pi_gamma = (self.E @ self.E.t()).pinverse()


        ## noise ##
        Sigma_w = torch.nn.functional.pad(Sigma_w, (0, 1, 0, 1))                # padding for matrix computations of augmented states (see function f)
        self.phi = phi                                                          # smoothness of temporal correlations
        self.z = noise(self.flag, self.T, self.dt, Sigma_z, self.n+1, self.e_sim, self.phi)
        self.w = noise(self.flag, self.T, self.dt, Sigma_w, self.n+1, self.e_sim, self.phi)
        self.v = noise(self.flag, self.T, self.dt, Sigma_v, self.n, self.e_sim, self.phi)

        self.C = symsqrt(self.w.Sigma)
        self.H = symsqrt(self.z.Sigma)
        
        self.Pi_w = self.w.Pi
        self.Pi_z = self.z.Pi
        self.Pi_v = self.v.Pi

        ## history ##
        self.history()

        # add padding for number of embedding orders                # FIXME: This needs to deal with the new format of our variables, now in matrices to be flatted/unflattend as needed
        self.addPadding()

        ## dimensions checks ##                                     # TODO: put this earlier, interrupt before padding?
        self.runChecks()





    def addPadding(self):
        if self.n > self.r:                                                     # for now, assuming always self.n >= self.r, if not, the noise needs to be adapted
            self.B_u = torch.nn.functional.pad(self.B_u, (0, 0, 0, self.n - self.r))
            self.B_u = torch.nn.functional.pad(self.B_u, (0, self.n - self.r, 0, 0))

            self.B_a = torch.nn.functional.pad(self.B_a, (0, 0, 0, self.n - self.r))
            self.B_a = torch.nn.functional.pad(self.B_a, (0, self.n - self.r, 0, 0))

            self.G = torch.nn.functional.pad(self.G, (0, 0, 0, self.n - self.r))
            self.G = torch.nn.functional.pad(self.G, (0, self.n - self.r, 0, 0))
        elif self.n < self.r:
            self.A = torch.nn.functional.pad(self.A, (0, 0, 0, self.r - self.n))
            self.A = torch.nn.functional.pad(self.A, (0, self.r - self.n, 0, 0))

            self.F = torch.nn.functional.pad(self.F, (0, 0, 0, self.r - self.n))
            self.F = torch.nn.functional.pad(self.F, (0, self.r - self.n, 0, 0))

            Sigma_w = torch.nn.functional.pad(Sigma_w, (0, self.r - self.n, 0, 0))
            Sigma_z = torch.nn.functional.pad(Sigma_z, (0, self.r - self.n, 0, 0))
            Sigma_v = torch.nn.functional.pad(Sigma_v, (0, self.r - self.n, 0, 0))

            print('Not fully implemented, quitting.')
            quit()

        # self.A = torch.nn.functional.pad(self.A, (0, 0, 0, 1))            # the matrix A must have an extra order since we are writing down n+1 variables in the vector x
        # self.A = torch.nn.functional.pad(self.A, (0, 1, 0, 0))

        # self.B_u = torch.nn.functional.pad(self.B_u, (0, 0, 0, 1))        # same
        # self.B_u = torch.nn.functional.pad(self.B_u, (0, 1, 0, 0))

        # self.B_a = torch.nn.functional.pad(self.B_a, (0, 0, 0, 1))        # same
        # self.B_a = torch.nn.functional.pad(self.B_a, (0, 1, 0, 0))

    def history(self):
        # TODO: this should be used with parsimony to avoid the RAM blowing up for big models
        self.y_history = torch.zeros(self.iterations, *self.y.shape, device = DEVICE)                                # TODO: remove this as it is now redundant with all the information in the original variables
        self.x_history = torch.zeros(self.iterations, *self.x.shape, device = DEVICE)
        self.u_history = torch.zeros(self.iterations, *self.u.shape, device = DEVICE)
        self.a_history = torch.zeros(self.iterations, *self.a.shape, device = DEVICE)
        self.eta_u_history = torch.zeros(self.iterations, *self.eta_u.shape, device = DEVICE)

        if self.flag == 'GP':
            self.w_history = self.z.noise
            self.z_history = self.w.noise

        self.F_history = torch.zeros(self.iterations, 1, device = DEVICE)

        self.eps_v_history = torch.zeros(*self.y_history.shape, device = DEVICE)
        self.eps_x_history = torch.zeros(*self.x_history.shape, device = DEVICE)
        self.eps_eta_history = torch.zeros(*self.u_history.shape, device = DEVICE)
        self.eps_theta_history = torch.zeros(self.iterations, *self.theta.shape, device = DEVICE)
        self.eps_gamma_history = torch.zeros(self.iterations, *self.gamma.shape, device = DEVICE)

        self.xi_v_history = torch.zeros(*self.eps_v_history.shape, device = DEVICE)
        self.xi_x_history = torch.zeros(*self.eps_x_history.shape, device = DEVICE)
        self.xi_eta_history = torch.zeros(*self.eps_eta_history.shape, device = DEVICE)
        self.xi_theta_history = torch.zeros(*self.eps_theta_history.shape, device = DEVICE)
        self.xi_gamma_history = torch.zeros(*self.eps_gamma_history.shape, device = DEVICE)

    def checkControl(self):
        if (len(self.B_u) != 0) or (len(self.B_a) != 0):
            if len(self.B_u) == 0:
                self.B_u = torch.zeros(self.r, self.r, device=DEVICE)           # inputs dimension (external dynamics)
            else:
                self.B_a = torch.zeros(self.r, self.r, device=DEVICE)           # input matrix (self-generated dynamics), default
        else:
            self.B_u = torch.zeros(self.r, self.r, device=DEVICE)               # input matrix (external dynamics), default
            self.B_a = torch.zeros(self.r, self.r, device=DEVICE)               # input matrix (self-generated dynamics), default
        
        if len(self.G) == 0:
            self.G = torch.zeros(self.r, self.r, device=DEVICE)                 # input matrix (observations), default

        if len(self.dyda) == 0:
            self.dyda = torch.zeros(self.sim*(self.e_sim+1), 1, device=DEVICE)  # input matrix (self-generated dynamics), default


    def runChecks(self):
        ### Run some simple checks on tensors dimensions
        # if self.A.size(0) != self.C.size(0):
        #     print('State transition matrix and process noise matrix sizes don\'t match. Please check and try again.')
        #     quit()
        if self.F.size(0) != self.H.size(0):
            print('Measurement matrix and measurement noise matrix sizes don\'t match. Please check and try again.')
            quit()

    def f(self, x, u, a):
        self.x = x                                                  # necessary because of the implementation of 'jacobian'/'hessian'
        self.u = u
        self.a = a

        # TODO: generalise this to include nonlinear treatments
        try:
            # aa = self.x.squeeze().unflatten(0, (self.e_sim+1, self.sim+1)).t()[:-1,:]
            # # print(self.x)
            # # aa = self.x[:-1, :].t().flatten().unsqueeze(1)
            # print(aa)
            # # aa = self.x.squeeze().unflatten(0, (self.sim+1, self.e_sim+1))
            # # print(aa)
            # aa = aa.t().flatten().unsqueeze(1)
            # print(aa)
            # aa = self.u[:-1, :].t().flatten().unsqueeze(1)
            # aa = self.a[:-1, :].t().flatten().unsqueeze(1)
            # aa =  (self.A @ self.x[:-1, :].t().flatten().unsqueeze(1) + self.B_u @ self.u[:-1, :].t().flatten().unsqueeze(1) + self.B_a @ self.a[:-1, :].t().flatten().unsqueeze(1)).squeeze().unflatten(0, (self.e_sim+1, self.sim)).t()                               # the extra order exluded in self.x.view(self.e_sim,self.sim+1)[:,:-1] was simply added to ensure consistency for n+1 embeddings given only n equations
            # bb = self.A @ self.x.squeeze().unflatten(0, (self.e_sim+1, self.sim+1)).t()[:-1,:].t().flatten().unsqueeze(1)
            # print(bb)
            # cc = self.x.flatten()
            # print(cc)
            # # print(aa.t().shape)
            # print(aa.squeeze().unflatten(0, (self.e_sim+1, self.sim)).t())

            xR = self.x.squeeze().unflatten(0, (self.e_sim+1, self.sim+1)).t()[:-1,:].t().flatten().unsqueeze(1)
            uR = self.u.squeeze().unflatten(0, (self.e_sim+1, self.sim+1)).t()[:-1,:].t().flatten().unsqueeze(1)
            aR = self.a.squeeze().unflatten(0, (self.e_sim+1, self.sim+1)).t()[:-1,:].t().flatten().unsqueeze(1)

            # Alternative (easier to debug?) below:
            # xR = self.x[Diff(Diff(self.x, self.sim+1, self.e_sim+1, shift=-1), self.sim+1, self.e_sim+1).nonzero(as_tuple=True)].unsqueeze(1)

            f = (self.A @ xR + self.B_u @ uR + self.B_a @ aR)

            fPadded = torch.nn.functional.pad(f.squeeze().unflatten(0, (self.e_sim+1, self.sim)).t(), (0,0,0,1))                 # without this extra padding, the Jacobian lacks dimensions since the output is otherwise based on a reduced state space

            result = fPadded.t().flatten().unsqueeze(1)

            return result
        except RuntimeError:
            print("Dimensions don't match!")
            return
    
    def g(self, x, u, a):
        self.x = x                                                  # necessary because of the implementation of 'jacobian'/'hessian'
        self.u = u
        self.a = a

        # TODO: generalise this to include nonlinear treatments
        try:
            return self.F @ self.x# + self.G @ self.u
        except RuntimeError:
            print("Dimensions don't match!")
            return

    def k_theta(self):                                                             # TODO: for simplicity we hereby assume that parameters are independent of other variables
        # TODO: generalise this to include nonlinear treatments
        try:
            return self.D @ self.eta_theta
        except RuntimeError:
            print("Dimensions don't match!")
            return
    
    def k_gamma(self):                                                             # TODO: for simplicity we hereby assume that hyperparameters are independent of other variables
        # TODO: generalise this to include nonlinear treatments
        try:
            return self.E @ self.eta_gamma
        except RuntimeError:
            print("Dimensions don't match!")
            return

    def prediction_errors(self, i):
        self.eps_v = self.y - self.g(self.x, self.u, self.a)
        self.eps_x = Diff(self.x, (self.sim+1), (self.e_sim+1)) - self.f(self.x, self.u, self.a)
        self.eps_eta = self.u - self.eta_u
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

    def free_energy(self, y, x, u, i):
        self.y = y
        self.x = x                                                  # necessary because of the implementation of 'jacobian'/'hessian' in torch.autograd.functional
        self.u = u

        self.prediction_errors(i)

        return .5 * (self.eps_v.t() @ self.xi_v + self.eps_x.t() @ self.xi_x + self.eps_eta.t() @ self.xi_eta + \
                            self.eps_theta.t() @ self.xi_theta + self.eps_gamma.t() @ self.xi_gamma + \
                            ((self.n + self.r + self.p + self.h) * torch.log(2*torch.tensor([[math.pi]])) - torch.logdet(self.Pi_z) - torch.logdet(self.Pi_w) - \
                            torch.logdet(self.Pi_theta) - torch.logdet(self.Pi_gamma)).sum())           # TODO: add more terms due to the variational Gaussian approximation?

    def step(self, i, method=1):
        # methods:
        # 0: Euler-Maruyama
        # 1: Local linearisation (Ozaki '92)

        # TODO: The following code works (?) for linear functions (and not nonlinear ones) in virtue of the fact that generalised coordinates for linear models are trivial; for nonlinear models, see snippet "from spm_ADEM_diff"
        # TODO: for nonlinear systems, higher embedding orders of y, x, v should contain derivatives of functions f and g

        self.updateGeneralisedCoordinates(i)

        if method == 0:
            self.dx = self.f(self.x, self.u, self.a) + self.C @ self.w.noise[i,:].unsqueeze(1)
            self.x = self.x + self.dt * self.dx
        elif method == 1:
            # inputs = (self.x, self.u, self.a)
            # print(inputs)
            self.J = self.df        # TODO: wait for a decent implementation of 'hessian' and 'jacobian' on all inputs similar to backward
            self.J_x = self.J[0].squeeze()                                                              # (at the moment both functions rely on grad, which requires specifying inputs). If not, to save some 
            self.J_a = self.J[2].squeeze()
            # print(self.J_x)
            # print(self.J_a)
            # print(self.f(self.x, self.u, self.a))
            # print(self.x)
            self.dx = dx_ll(self.dt, self.J_x + self.J_a, Diff(self.x, self.sim+1, self.e_sim+1))                # FIXME: I believe this line is not taking into account 'a' properly, please check Jacobian J_x
            # print(Diff(self.x, self.sim+1, self.e_sim+1))
            # print(self.J_x + self.J_a)
            # print(self.x)
            # self.du = (torch.matrix_exp(self.dt * self.J_u) - torch.eye(self.sim)) @ self.J_u.pinverse() @ ??????                         # TODO: should we implement a way to give dynamic equations for inputs too?
            # self.x.squeeze().unflatten(0, (self.e_sim+1, self.sim+1)).t()[1:,:].t().flatten().unsqueeze(1) = self.dx.squeeze().unflatten(0, (self.e_sim+1, self.sim+1)).t()[1:,:].t().flatten().unsqueeze(1)

            # print(self.dx)
            self.x = self.x + self.dt * self.dx
            # GM.u = GM.u + dt * du
        else:
            print('Method not implemented. Please check and try a different method.')
            quit()
        
        # for j in range(self.e_sim+1):                                                       # In a dynamic model with x, x', x'', x''', ..., the last variable does not get updated during integration, i.e., \dot{x} => x = x + x' = x + f(x)
        #     self.x[self.sim*(j+1)-1] = self.dx[self.sim*(j+1)-2]                            # \dot{x'} => x' = x' + x'' = x' + f(x') but x'' = f(x') (there is no x'' = x'' + x''' = x'' + f(x'') equation)
        self.y = self.g(self.x, self.u, self.a) + self.H @ self.z.noise[i,:].unsqueeze(1)

    def updateGeneralisedCoordinates(self, i):
        ### from spm_ADEM_diff
        # u.v{1}  = spm_vec(vi);
        # u.x{2}  = spm_vec(fi) + u.w{1};
        # for i = 2:(n - 1)
        #     u.v{i}     = dg.dv*u.v{i} + dg.dx*u.x{i} + dg.da*u.a{i} + u.z{i};
        #     u.x{i + 1} = df.dv*u.v{i} + df.dx*u.x{i} + df.da*u.a{i} + u.w{i};
        # end
        # u.v{n}  = dg.dv*u.v{n} + dg.dx*u.x{n} + dg.da*u.a{n} + u.z{n};

        # f = self.f(self.x, self.u, self.a)
        # g = self.g(self.x, self.u, self.a)

        inputs = (self.x, self.u, self.a)
        self.dg = torch.autograd.functional.jacobian(lambda x, u, a: self.g(x, u, a), inputs)
        self.df = torch.autograd.functional.jacobian(lambda x, u, a: self.f(x, u, a), inputs)

        self.dgdx = self.dg[0].squeeze()[0:self.sim+1, 0:self.sim+1]
        self.dgdu = self.dg[1].squeeze()[0:self.sim+1, 0:self.sim+1]
        self.dgda = self.dg[2].squeeze()[0:self.sim+1, 0:self.sim+1]
        self.dfdx = self.df[0].squeeze()[0:self.sim, 0:self.sim]
        self.dfdu = self.df[1].squeeze()[0:self.sim, 0:self.sim]
        self.dfda = self.df[2].squeeze()[0:self.sim, 0:self.sim]

        # print(self.x[1:self.sim+1])
        # print(self.f(self.x, self.u, self.a)[:self.sim])
        # print((self.C @ self.w.noise[i,:].unsqueeze(1))[:self.sim])

        print(self.x)
        print(self.f(self.x, self.u, self.a)[:(self.sim+1)-1].shape)
        print(self.f(self.x, self.u, self.a)[:(self.sim+1)-1])

        self.x[1:(self.sim+1)] = self.f(self.x, self.u, self.a)[:(self.sim+1)-1]# + (self.C @ self.w.noise[i,:].unsqueeze(1))[:(self.sim+1)-1]
        self.y[0:(self.sim+1)] = self.g(self.x, self.u, self.a)[:(self.sim+1)] + (self.H @ self.z.noise[i,:].unsqueeze(1))[:(self.sim+1)]

        # print(self.x)

        for j in range(1,self.e_sim+1):
            # a = self.x[j*(self.sim+1)+1:(j+1)*(self.sim+1)]
            # print(a)
            # aa = self.x[j*(self.sim+1):(j+1)*(self.sim+1)-1]
            # print(aa)
            # aaa = dfdx @ self.x[j*(self.sim+1):(j+1)*(self.sim+1)-1]
            # print(dfdx)
            print(self.x)
            print(self.x[j*(self.sim+1)+1:(j+1)*(self.sim+1)].shape)
            self.x[j*(self.sim+1)+1:(j+1)*(self.sim+1)] = self.dfdx @ self.x[j*(self.sim+1):(j+1)*(self.sim+1)-1] + self.dfdu @ self.u[j*(self.sim+1):(j+1)*(self.sim+1)-1] + self.dfda @ self.a[j*(self.sim+1):(j+1)*(self.sim+1)-1]# + (self.C @ self.w.noise[i,:].unsqueeze(1))[j*(self.sim+1):(j+1)*(self.sim+1)-1]
            print(self.x)
            self.y[j*(self.sim+1):(j+1)*(self.sim+1)] = self.dgdx @ self.x[j*(self.sim+1):(j+1)*(self.sim+1)] + self.dgdu @ self.u[j*(self.sim+1):(j+1)*(self.sim+1)] + self.dgda @ self.a[j*(self.sim+1):(j+1)*(self.sim+1)] + (self.H @ self.z.noise[i,:].unsqueeze(1))[j*(self.sim+1):(j+1)*(self.sim+1)]
            # print(self.y)
        # print(self.y[(self.e_sim)*(self.sim+1):])
        self.y[(self.e_sim)*(self.sim+1):] = self.dgdx @ self.x[(self.e_sim)*(self.sim+1):] + self.dgdu @ self.u[(self.e_sim)*(self.sim+1):] + self.dgda @ self.a[(self.e_sim)*(self.sim+1):] + (self.H @ self.z.noise[i,:].unsqueeze(1))[(self.e_sim)*(self.sim+1):]
        # print(self.x)
        # print(self.y)


    
    def setObservations(self, y):                                                                   # TODO: check if there's a more elegant way of doing this
        self.y = y
        # .detach()
        # self.y.requires_grad = True
    
    def saveHistoryVariables(self, i):                                                              # TODO: remove
        self.y_history[i,:] = self.y
        self.x_history[i,:] = self.x
        self.u_history[i,:] = self.u
        self.a_history[i,:] = self.a
        self.eta_u_history[i,:] = self.eta_u

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

    def hook(self, grad):
        print(grad)
        self.h.remove()