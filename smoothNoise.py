import math
import torch
from functions import kronecker, symsqrt
from scipy.linalg import toeplitz
from globalVariables import DEVICE, _small, _large

class noise():                                                  # TODO: In the future, make this more 'OO-friendly', so far the methods are just direct translations of their spm counterparts for easier comparisons
    def __init__(self, flag, T, dt, Sigma, n, e_n, phi):
        
        # n: number of variables
        # e_n: embedding orders
        # S: smoothness, R: roughness matrices
        # Sigma: covariance noise
        # Pi: precision noise

        self.T = T
        self.dt = dt
        self.iterations = int(T/dt)

        self.n = n
        self.e_n = e_n
        self.phi = phi

        if len(Sigma) > 0:
            self.S_inv = self.spm_DEM_R(self.e_n+1, self.phi)                                           # temporal precision matrix observation noise, roughness
            self.Sigma = kronecker(self.S_inv, Sigma)                                                   # covariance matrix observation noise including higher embedding orders # TODO: find a torch based version of the krocker product
            self.Pi = self.Sigma.pinverse()                                                             # precision matrix observation noise including higher embedding orders

            if flag == 'GP':
                self.noiseSmoothened = self.spm_DEM_z(self.n, self.phi, self.T, self.dt)
                self.noise = torch.zeros(self.iterations, self.n*(self.e_n+1), device=DEVICE)
                for i in range(self.iterations):
                    self.noise[i, :] = self.spm_DEM_embed(self.noiseSmoothened, self.e_n+1, i)          # FIXME: This I don't fully understand, but if we impose dt < 1. here we get a weird behaviour, e.g., dt = 0.1 only the first 1/10 of the sequence is considered and then the noise is flat
                                                                                                        # FIXME: After chacking the above, find out if the precisions needs to be changed, following equation 55 of the DEM paper. So far no hint in the code, but maybe in spm_DEM_R?
        else:
            self.Sigma = _large * torch.ones(self.n*(self.e_n+1), self.n*(self.e_n+1), device=DEVICE)
            self.Pi = self.Sigma.pinverse()
            self.z = torch.zeros(self.n*(self.e_n+1), 1, device=DEVICE)


    def spm_DEM_R(self, n, s):
        # function adapted and simplied from SPM, original description below
        # TODO: include different correlations if necessary, for now only Gaussian

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



    def spm_DEM_embed(self, Y, n, t, dt=1, d=0):
        # function adapted and nplied from SPM, original description below

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
            y[:,i] = Y[:,k-1] @ E[i,:].t()
        
        z = y.t().flatten().t()                     # TODO: Look for more elegant solution (all noise, different variables and embedding orders in one column [y1 y2 ... yn, y'1 y'2 ... y'n, y''1 y''2 ... y''n]^T)

        return z




    def spm_DEM_z(self, n, s, T, dt=1.):
        # see also https://www.kaggle.com/charel/learn-by-example-active-inference-noise/comments

        t = torch.arange(0, T, dt)                                          # autocorrelation lags
        K = torch.from_numpy(toeplitz(torch.exp(-t**2/(2*s**2))))           # convolution matrix
        K = torch.diag(1./torch.sqrt(torch.diag(K @ K.t()))) @ K

        noise = torch.randn(n, int(T/dt)) @ K
        
        return noise