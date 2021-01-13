from functions import *

class freeEnergyFunctional():
    def __init__(self, A, B_u, B_a, F, G, D, E, y, x, u, a, eta_theta, Pi_z, Pi_w, Pi_v, Pi_theta, Pi_gamma):
        self.A = A
        self.B_u = B_u
        self.B_a = B_a
        self.F = F
        self.G = G
        self.D = D
        self.E = E
        self.y = y
        self.x = x
        self.u = u
        self.a = a
        self.eta_theta = eta_theta

    def prediction_errors(self, i):
        self.eps_v = self.y - self.g(i)
        self.eps_x = Diff(self.x, self.n, self.e_sim+1) - self.f(i)
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

    def free_energy(self, i):
        self.prediction_errors(i)

        return .5 * (self.eps_v.t() @ self.xi_v + self.eps_x.t() @ self.xi_x + self.eps_eta.t() @ self.xi_eta + \
                            self.eps_theta.t() @ self.xi_theta + self.eps_gamma.t() @ self.xi_gamma + \
                            ((self.n + self.r + self.p + self.h) * torch.log(2*torch.tensor([[math.pi]])) - torch.logdet(self.Pi_z) - torch.logdet(self.Pi_w) - \
                            torch.logdet(self.Pi_theta) - torch.logdet(self.Pi_gamma)).sum())           # TODO: add more terms due to the variational Gaussian approximation?