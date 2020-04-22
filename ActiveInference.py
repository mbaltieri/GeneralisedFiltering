#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 3 2019

Simulation of affordances in active inference

@author: manuelbaltieri
"""
#%%
from Functions import *

np.random.seed(42)

### define font size for plots ###
#
SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE)            # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)       # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)        # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)       # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)       # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)       # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)     # fontsize of the figure title
#

dt = .01

gamma = 1                                                   # drift in OU process (if you want to simulate coloured noise)
plt.close('all')
small_value = np.exp(-50)
MAX_TEMPORAL_ORDERS = 20

# n-th derivative of rho function
# define derivatives up to MAX_TEMPORAL_ORDERS

# parameters for environment (generative process)
# obs_states = 4
# hidden_states = 2                                           # x, in Friston's work
# hidden_causes = 3                                           # v, in Friston's work

embed_orders_obs = 2
embed_orders_states = 2                                      # generalised coordinates for hidden states x, but only using n-1
embed_orders_causes = 2                                      # generalised coordinates for hidden causes v (or \eta in biorxiv manuscript), but only using n-1

# parameters for agent (generative model)
# obs_states_gm = 4
# hidden_states_gm = 4                                           # x, in Friston's work
# hidden_causes_gm = 2                                           # v, in Friston's work

embed_orders_obs_gm = 2
embed_orders_states_gm = 2                                      # generalised coordinates for hidden states x, but only using n-1
embed_orders_causes_gm = 2                                      # generalised coordinates for hidden causes v (or \eta in biorxiv manuscript), but only using n-1

delta = 1.50                                                # parameters simulating a simple mass-spring system as the environment
epsilon = 15.0

phi = 2.                                                    # smoothness of temporal correlations

gamma_z = -3
gamma_w = -3

sigma_z = np.exp(-gamma_z)
sigma_w = np.exp(-gamma_w)

def FreeEnergy(y, mu_x, mu_v, mu_pi_z, mu_pi_w, A_gm, B_gm, F_gm):
    return .5 * (np.sum(np.dot(np.dot((y - np.dot(F_gm, mu_x)).transpose(), mu_pi_z), (y - np.dot(F_gm, mu_x)))) + \
                np.sum(np.dot(np.dot((mu_x[:, 1:] - f_gm(mu_x[:, :-1], mu_v[:, :-1], A_gm, B_gm)).transpose(), mu_pi_w), (mu_x[:, 1:] - f_gm(mu_x[:, :-1], mu_v[:, :-1], A_gm, B_gm)))) - \
                np.trace(np.log(mu_pi_z * mu_pi_w)))


def doubleIntAI(simulation, iterations, generative_process, generative_model):
    # environment variables (to be reset for each simulation)

    obs_states = generative_process.obs_states
    hidden_states = generative_process.hidden_states
    hidden_causes = generative_process.hidden_causes

    obs_states_gm = generative_model.obs_states
    hidden_states_gm = generative_model.hidden_states
    hidden_causes_gm = generative_model.hidden_causes

    x = np.zeros((hidden_states, 1))           # position
    
    v = np.zeros((hidden_causes, 1))
    y = np.zeros((obs_states, 1))
    eta = np.zeros((hidden_causes, embed_orders_states - 1))

    A_tilde = np.kron(np.eye(embed_orders_states), generative_process.A)
    B_tilde = np.kron(np.eye(embed_orders_causes), generative_process.B)
    C_tilde = np.kron(np.eye(embed_orders_states), generative_process.C)
    D_tilde = np.kron(np.eye(embed_orders_obs), generative_process.D)
    F_tilde = np.kron(np.eye(embed_orders_obs), generative_process.F)
    G_tilde = np.kron(np.eye(embed_orders_causes), generative_process.G)

    # define temporal precision matrix
    S = smoothnessMatrix(3, phi)

    ### free energy variables
    # parameters for generative model

    A_tilde_gm = np.kron(np.eye(embed_orders_states_gm), generative_model.A)
    B_tilde_gm = np.kron(np.eye(embed_orders_causes_gm), generative_model.B)
    F_tilde_gm = np.kron(np.eye(embed_orders_states_gm), generative_model.F)
    G_tilde_gm = np.kron(np.eye(embed_orders_causes_gm), generative_model.G)
    
    # actions
    a = np.zeros((1, embed_orders_states - 1))
    
    # states
    mu_x = np.zeros((hidden_states_gm, 1))
    
    # inputs
    mu_v = np.zeros((hidden_causes_gm, 1))
    
    # minimisation variables and parameters
    dFdmu_x = np.zeros((hidden_states_gm, embed_orders_states_gm))
    Dmu_x = np.zeros((hidden_states_gm, embed_orders_states_gm))
    
    k_mu_x = 1                                                  # learning rate perception
    k_mu_v = 1
    k_a = np.exp(2)                                                     # learning rate action
    
    # noise on sensory input (world - generative process)
    # gamma_z = 0 * np.ones((obs_states, obs_states))    # log-precisions
    #gamma_z[:,1] = gamma_z[:,0] - np.log(2 * gamma)
    # pi_z = np.zeros((obs_states, obs_states))
    # np.fill_diagonal(D, np.exp(-gamma_z/2))
    # sigma_z = np.linalg.inv(splin.sqrtm(D))
    # np.fill_diagonal(D, sigma_z)


    z = np.random.randn(iterations, obs_states)
    S_z = smoothnessMatrix(embed_orders_obs, phi)
    Pi_z_tilde = np.kron(S_z, np.linalg.pinv(np.dot(D, D.transpose())))
    Sigma_z_tilde = np.linalg.pinv(Pi_z_tilde)

    # z_tilde = np.kron(S, z)
    # z_tilde1 = np.kron(z, S)
    
    # noise on motion of hidden states (world - generative process)
    # gamma_w = 2                                                  # log-precision
    # pi_w = np.zeros((hidden_states, hidden_states))
    # np.fill_diagonal(C, np.exp(-gamma_w/2))
    # sigma_w = np.linalg.inv(splin.sqrtm(C))
    # np.fill_diagonal(C, sigma_w)

    w = np.random.randn(iterations, hidden_states)
    S_w = smoothnessMatrix(embed_orders_states, phi)
    Pi_w_tilde = np.kron(S_w, np.linalg.pinv(np.dot(C, C.transpose())))
    Sigma_w_tilde = np.linalg.pinv(Pi_w_tilde)

    # w_tilde = np.kron(S, w)
    
    
    # agent's estimates of the noise (agent - generative model)
    # mu_gamma_z = 3 * np.identity((obs_states_gm))    # log-precisions
    # if obs_states > 1:
    #     for i in range(1,obs_states_gm):
    #         mu_gamma_z[i, i] = mu_gamma_z[i-1, i-1] - np.log(2 * gamma)     # higher embedding orders are correlated, although this is wrong for more than 2 orders
    # mu_gamma_z[1, 1] = mu_gamma_z[0, 0] - np.log(2 * gamma)
    # mu_gamma_z[2, 2] = mu_gamma_z[1, 1] - np.log(2 * gamma)
    # mu_pi_z = np.exp(mu_gamma_z) * np.identity((obs_states_gm))
    # print(mu_pi_z)

    mu_gamma_z = 0                                                          # log-precision
    Mu_sigma_z = np.exp(-mu_gamma_z) * np.identity((obs_states_gm))
    Mu_pi_z = np.dot(Mu_sigma_z, Mu_sigma_z.transpose())
    S = smoothnessMatrix(embed_orders_obs_gm, phi)
    Mu_pi_z_tilde = np.kron(S, np.linalg.pinv(Mu_pi_z))
    # print(Mu_pi_z_tilde)


    
    mu_gamma_w = -10 * np.identity((hidden_states_gm))   # log-precision
    if obs_states > 1:
        for i in range(1,hidden_states_gm):
            mu_gamma_w[i, i] = mu_gamma_w[i-1, i-1] - np.log(2 * gamma)        
    # mu_gamma_w[1, 1] = mu_gamma_w[0, 0] - np.log(2 * gamma)
    # mu_gamma_w[2, 2] = mu_gamma_w[1, 1] - np.log(2 * gamma)
    mu_pi_w = np.exp(mu_gamma_w) * np.identity((hidden_states_gm))

    mu_gamma_w = -10                                                          # log-precision
    Mu_sigma_w = np.exp(-mu_gamma_w) * np.identity((hidden_states_gm))
    Mu_pi_w = np.dot(Mu_sigma_w, Mu_sigma_w.transpose())
    S = smoothnessMatrix(embed_orders_states_gm, phi)
    Mu_pi_w_tilde = np.kron(S, np.linalg.pinv(Mu_pi_w))
    # print(Mu_pi_w_tilde)

    x = 10 * np.random.rand(hidden_states, 1) - 5
    # S = smoothnessMatrix(embed_orders_states, phi)
    # x = np.kron(S, x)
    x_tilde = np.kron(np.eye(embed_orders_states), x)               # FIXME: for nonlinear systems, higher embedding orders should contain derivatives of functions f and g

    # S = smoothnessMatrix(embed_orders_causes, phi)
    # v = np.kron(S, v)
    v_tilde = np.kron(np.eye(embed_orders_causes), v)

    # S = smoothnessMatrix(embed_orders_obs, phi)
    # y = np.kron(S, y)
    y = 10 * np.random.rand(obs_states, 1) - 5
    y_tilde = np.kron(np.eye(embed_orders_obs), y)

    # S = smoothnessMatrix(embed_orders_states_gm, phi)
    # mu_x = np.kron(S, mu_x)
    mu_x_tilde = np.kron(np.eye(embed_orders_states_gm), mu_x)

    mu_v_tilde = np.kron(np.eye(embed_orders_causes_gm), mu_v)
    
    # if the initialisation is too random, then this agent becomes ``disillusioned''
    # mu_x[0, 0] = x[0, 0] + .1*np.random.randn()
    # mu_x[1, 0] = x[0, 1] + .1*np.random.randn()
    # mu_x[0, 1] = mu_x[1, 0]
    
    # automatic differentiation
    dFdobs = grad(FreeEnergy, 0)
    dFdmu_states = grad(FreeEnergy, 1)
    dFdmu_inputs = grad(FreeEnergy, 2)
    
    # history
    y_history = np.zeros((iterations, obs_states_gm, embed_orders_states_gm))
    x_history = np.zeros((iterations, hidden_states, embed_orders_states))
    psi_history = np.zeros((iterations, obs_states_gm, embed_orders_states_gm - 1))
    mu_x_history = np.zeros((iterations, hidden_states_gm, embed_orders_states_gm))
    a_history = np.zeros((iterations, hidden_causes_gm, embed_orders_states_gm))
    FE_history = np.zeros((iterations,))
    mu_v_history = np.zeros((iterations, hidden_causes_gm, embed_orders_states_gm))

    y_tilde_history = np.zeros((iterations, len(y_tilde), len(y_tilde[0])))
    x_tilde_history = np.zeros((iterations, len(x_tilde), len(x_tilde[0])))
    psi_history = np.zeros((iterations, obs_states_gm, embed_orders_states_gm - 1))
    mu_x_tilde_history = np.zeros((iterations, len(mu_x_tilde), len(mu_x_tilde[0])))
    a_history = np.zeros((iterations, hidden_causes_gm, embed_orders_states_gm))
    FE_history = np.zeros((iterations,))
    mu_v_tilde_history = np.zeros((iterations, len(mu_v_tilde), len(mu_v_tilde[0])))

    w_tilde = np.zeros((iterations, hidden_states*embed_orders_states_gm, embed_orders_states_gm))
    for i in range(iterations):
        w_tilde[i,:,:] = Sigma_w_tilde @ np.kron(np.eye(embed_orders_states_gm),w[i,:, None])

    z_tilde = np.zeros((iterations, obs_states*embed_orders_obs, embed_orders_obs))
    for i in range(iterations):
        z_tilde[i,:,:] = Sigma_z_tilde @ np.kron(np.eye(embed_orders_obs),z[i,:, None])
    
    # w_tilde = Sigma_w_tilde @ np.kron(np.eye(embed_orders_states_gm),w[0,:, None])
    # z_tilde = Sigma_z_tilde @ np.kron(np.eye(embed_orders_obs),z[0,:, None])


    for i in range(iterations - 1):


        # state-dependend precisions
        # eta = k_1 - gamma_1 * sigmoid(mu_x[0, 0] + mu_v[0, 0] - mu_v[1, 0])
        # phi = k_2 - gamma_2 * sigmoid(mu_x[2, 0] - mu_v[0, 0] + mu_v[1, 0])

        # v[1, 0] = 20
        # a = np.sin(2*i /iterations * 2*np.pi)*10
        # if x[0,0] <= 5 or x[0,0] >= 100:
        #     v[2,0] = 80
        # else:
        #     v[2,0] = 0
        
        # mu_x_history[i, :, :] = mu_x                # save it at the very beginning since the first jump is rather quick
        
        # y[:, :] = getObservation(x, v, np.dot(np.dot(C, sigma_w), w[i, :]))
        # y[:, :] = getObservation(x, v, w[i, :])
        # y[2, 0] = y[1, 1]                           # manually assign the acceleration as observed by the agent
        
        # psi = y[:,:-1] + np.dot(np.dot(D, sigma_z), z[i, :, None])

        f = lambda x, v, A, B: A @ x + B @ v
        
        dx_tilde = f(x_tilde, v_tilde, A_tilde, B_tilde)
        dx_tilde += + w_tilde[i,:,:]
        x_tilde += dt * dx_tilde                        # FIXME: please check that the white noise is integrated appropriately

        g = lambda x, v, F_tilde, G_tilde: F_tilde @ x + G_tilde @ v
        
        y_tilde = g(x_tilde, v_tilde, F_tilde, G_tilde)
        y_tilde += z_tilde[i,:,:]

        y_history[i, :] = y
        x_history[i, :, :] = x

        ### minimise free energy ###
        # perception
        dFdmu_x_tilde = dFdmu_states(y_tilde, mu_x_tilde, mu_v_tilde, Mu_pi_z_tilde, Mu_pi_w_tilde, A_tilde_gm, B_tilde_gm, F_tilde_gm)
        Dmu_x_tilde = Diff(mu_x_tilde, hidden_states_gm, embed_orders_states_gm)

        dFdmu_v = dFdmu_inputs(y_tilde, mu_x_tilde, mu_v_tilde, Mu_pi_z_tilde, Mu_pi_w_tilde, A_tilde_gm, B_tilde_gm, F_tilde_gm)
        Dmu_v = Diff(mu_v_tilde, hidden_causes_gm, embed_orders_causes_gm)
        
        # action
        # dFdy = np.dot(mu_pi_z, (psi - mu_x[:, :-1]))
        dFdy = dFdobs(y_tilde, mu_x_tilde, mu_v_tilde, Mu_pi_z_tilde, Mu_pi_w_tilde, A_tilde_gm, B_tilde_gm, F_tilde_gm)
        dyda = np.ones((obs_states, embed_orders_states - 1))

        # noise
        dw = Diff(w_tilde[i,:,:], hidden_states, embed_orders_states)
        dz = Diff(z_tilde[i,:,:], obs_states, embed_orders_obs)
        
        # # save history
        y_tilde_history[i, :] = y_tilde
        x_tilde_history[i, :, :] = x_tilde[:, :]
        mu_x_tilde_history[i, :, :] = mu_x_tilde
        a_history[i] = a
        mu_v_tilde_history[i] = mu_v_tilde
        
        FE_history[i] = FreeEnergy(y_tilde, mu_x_tilde, mu_v_tilde, Mu_pi_z_tilde, Mu_pi_w_tilde, A_tilde_gm, B_tilde_gm, F_tilde_gm)
        
        
        # update equations
        mu_x_tilde += dt * k_mu_x * (Dmu_x_tilde - dFdmu_x_tilde)       # FIXME: Dmu_x_tilde doesn't work, there shouldn't be any dependences between x1 and x2, but only among generalised coordinates
        # mu_v += dt * k_mu_v * (Dmu_v - dFdmu_v)
        # mu_x += dt * k_mu_x * (- dFdmu_x)
        # mu_v += dt * k_mu_v * (- dFdmu_v)
        # mu_v[0, 0] = np.sin(i*2/iterations * 2*np.pi)*100
        # a += dt * - k_a * dyda.transpose().dot(dFdy)

        w_tilde[i,1::hidden_states,:] = w_tilde[i+1,1::hidden_states,:]
        # z_tilde[i,obs_states-1::obs_states,:] = z_tilde[i+1,obs_states-1::obs_states,:]       # FIXME: does this noise need to be integrated?

        w_tilde[i+1,:,:] = w_tilde[i,:,:] + np.sqrt(dt)*dw
        # z_tilde[i+1,:,:] = z_tilde[i,:,:] + np.sqrt(dt)*dz
        
        v[0, 0] = a
        
    
    return y_tilde_history, x_tilde_history, mu_x_tilde_history, a_history, mu_v_tilde_history, FE_history

simulation = 0
# 0: high spring stifness, strong damping
# 1: intermediate spring stifness, intermediate damping
# 2: low spring stifness, weak damping
# 3: as in simulation 0, but now we introduce an external force not modeled by the agent

T = 15
iterations = int(T / dt)

# generative process

ga = 9.81
d = 0.
theta_0 = 20.                                # constant flyball angle, static equilibrium, **desired engine speed**
Omega_0 = 1.                                # constant engine speed, static equilibrium
l = 10                                     # length of the arms
b = 2                                       # friction coefficient
m = b/2                                       # flyballs mass

k = 1                                       # "constant relaring flyball height and engine torque"
I = 1                                       # shaft torque


A = np.array([[0, 1, 0], [-ga*np.sin(theta_0)**2/(l*np.cos(theta_0)), -b/m, 2*ga*np.sin(theta_0)/l], [-k/(I*Omega_0), 0, 0]])
B = np.array([[0, 0, 0], [1, 0, 0]])                        # input matrix
C = sigma_w * np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])               # noise dynamics matrix
D = sigma_z * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])             # noise observations matrix
F = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])               # measurement matrix
G = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0], [0, 0, 1]])

# generative model
alpha_1 = 10
alpha_2 = 1
beta_1 = 1
beta_2 = 1

A_gm = np.array([[- alpha_1, 0, 0, 0], [0, - alpha_2, 0, 0], [0, 0, - beta_1, 0], [0, 0, 0, beta_2]])               # state transition matrix
B_gm = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])                     # input matrix
F_gm = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])               # measurement matrix
G_gm = np.zeros((4, 2))

class model():
    def __init__(self, A, F, B=0, C=0, G=0, D=0):
        self.A = A
        self.B = B
        self.C = C
        self.F = F
        self.G = G
        self.D = D

        self.obs_states = len(F)
        self.hidden_states = len(A)
        if np.ndim(B) > 0:
            self.hidden_causes = len(B[0])
        else:
            self.hidden_causes = 1

generative_model = model(A_gm, F_gm, B=B_gm, G=G_gm)
generative_process = model(A, F, C=C, D=D)

simulations_n = 1
y_history = np.zeros((simulations_n, iterations, generative_process.obs_states*embed_orders_obs, embed_orders_states_gm))
x_history = np.zeros((simulations_n, iterations, generative_process.hidden_states*embed_orders_states, embed_orders_states))
mu_x_history = np.zeros((simulations_n, iterations, generative_model.hidden_states*embed_orders_states_gm, embed_orders_states_gm))
a_history = np.zeros((simulations_n, iterations, generative_model.hidden_causes, embed_orders_states_gm))
mu_v_history = np.zeros((simulations_n, iterations, generative_model.hidden_causes*embed_orders_causes_gm, embed_orders_states_gm))
FE_history = np.zeros((simulations_n, iterations,))

# plt.figure(figsize=(9, 6))
# if simulation == 0:
#     plt.title('Double integrator - active inference \n no efference copy')
# elif simulation == 1:
#     plt.title('Double integrator - active inference \n no efference copy, weaker spring/damper')
# elif simulation == 2:
#     plt.title('Double integrator - active inference \n no efference copy, (much) weaker spring/damper')
# elif simulation == 3:
#     plt.title('Double integrator - active inference \n no knowledge of external force')
# plt.xlabel('Position ($m$)')
# plt.ylabel('Velocity ($m/s$)')
for k in range(simulations_n):
    y_history[k,:,:,:], x_history[k,:,:,:], mu_x_history[k,:,:,:], a_history[k,:,:,:], mu_v_history[k,:,:,:], FE_history[k,:] = doubleIntAI(simulation, iterations, generative_process, generative_model)
    # y_history[k,:,:,:], x_history[k,:,:,:] = doubleIntAI(simulation, iterations)


#     plt.plot(psi_history[k,:-1, 0, 0], psi_history[k,:-1, 1, 0], 'b')
#     plt.plot(mu_x_history[k, :-1, 0, 0], mu_x_history[k, :-1, 1, 0], 'r')
#     plt.plot(psi_history[k, 0, 0, 0], psi_history[k, 0, 1, 0], 'o', markersize = 15, label='Agent ' + str(k+1))
# plt.legend(loc=1)

# plt.figure()
# plt.plot(FE_history[0, :])
for k in range(simulations_n):
    fig = plt.figure(figsize=(10, 13))
    for i in range(generative_process.obs_states*embed_orders_obs):
        for j in range(embed_orders_obs):
            plt.subplot(generative_process.obs_states*embed_orders_obs,embed_orders_obs,j+1+i*embed_orders_obs)
            plt.plot(y_history[k,:-1, i, j], 'b')
            plt.plot(mu_x_history[k,:-1, i, j], 'r')
            plt.plot(x_history[k,:-1, int(i//2), j], 'k')


# plt.subplot(2,4,1)
# plt.plot(y_history[0,:-1, 0, 0], 'b')
# plt.plot(mu_x_history[0,:-1, 0, 0], 'r')
# plt.plot(x_history[0,:-1, 0, 0], 'k')
# plt.subplot(2,4,2)
# plt.plot(y_history[0,:-1, obs_states, 1], 'b')
# plt.plot(mu_x_history[0,:-1, obs_states, 1], 'r')
# plt.plot(x_history[0,:-1, 0, 1], 'k')
# plt.subplot(2,4,3)
# plt.plot(y_history[0,:-1, 1, 0], 'b')
# plt.plot(mu_x_history[0,:-1, 1, 0], 'r')
# plt.plot(x_history[0,:-1, 0, 0], 'k')
# plt.subplot(2,4,4)
# plt.plot(y_history[0,:-1, obs_states+1, 1], 'b')
# plt.plot(mu_x_history[0,:-1, hidden_states+1, 1], 'r')
# plt.plot(x_history[0,:-1, 0, 1], 'k')
# plt.subplot(2,4,5)
# plt.plot(y_history[0,:-1, 2, 0], 'b')
# plt.plot(mu_x_history[0,:-1, 2, 0], 'r')
# plt.plot(x_history[0,:-1, 0, 0], 'k')
# plt.subplot(2,4,6)
# plt.plot(y_history[0,:-1, obs_states+2, 1], 'b')
# plt.plot(mu_x_history[0,:-1, hidden_states, 1], 'r')
# plt.plot(x_history[0,:-1, 0, 1], 'k')
# plt.subplot(2,4,7)
# plt.plot(y_history[0,:-1, 3, 0], 'b')
# plt.plot(mu_x_history[0,:-1, 3, 0], 'r')
# plt.plot(x_history[0,:-1, 0, 0], 'k')
# plt.subplot(2,4,8)
# plt.plot(y_history[0,:-1, obs_states+3, 1], 'b')
# plt.plot(mu_x_history[0,:-1, hidden_states+1, 1], 'r')
# plt.plot(x_history[0,:-1, 0, 1], 'k')

# plt.figure()
# plt.plot(mu_v_history[0,:-1, 0, 0], 'b', label='a')
# plt.plot(mu_v_history[0,:-1, 1, 0], 'r', label='d')
# plt.plot(a_history[0, :-1, 0, 0], 'k', label='a (real)')
# plt.legend()


# plt.plot(a_history[0, :, 0, 0])

# plt.figure(figsize=(9, 6))
# if simulation == 0:
#     plt.title('Action of double integrator - active inference \n no efference copy')
# elif simulation == 1:
#     plt.title('Action of double integrator - active inference \n no efference copy, weaker spring/damper')
# elif simulation == 2:
#     plt.title('Action of double integrator - active inference \n no efference copy, (much) weaker spring/damper')
# elif simulation == 3:
#     plt.title('Action of double integrator - active inference \n no knowledge of external force')
# plt.xlabel('Time ($s$)')
# plt.ylabel('Action a ($m/s^2$)')
# for k in range(simulations_n):
#     plt.plot(np.arange(0, T-dt, dt), a_history[k,:-1,1,0], label='Agent ' + str(k+1))
# if simulation == 3:
#     plt.plot(np.arange(0, T-dt, dt), v_history[2,:-1,1,0], 'k', label='Ext. force')
# plt.xlim(0, T)
# plt.ylim(-250, 500)
# if simulation == 3:
#     plt.xticks(np.arange(0, T+1, 2))
# else:
#     plt.xticks(np.arange(0, T+1, 1))
# plt.legend(loc=1)

plt.show()
