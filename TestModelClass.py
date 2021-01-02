from layer import *
import functions
import matplotlib.pyplot as plt

# torch.autograd.set_detect_anomaly(True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device in use:", DEVICE)

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)

dt = .01
T = 20
iterations = int(T/dt)

l = 3                               # number of layers

# m = 2                               # observations dimension
# n = 2                               # hidden states dimension
# r = 2                               # inputs dimension
# p = 6                               # parameters dimension
# h = 3                               # hyperparameters dimension

e_n = 0                             # embedding dimension hidden states
e_r = 0                             # embedding dimension inputs
e_p = 0                             # embedding dimension parameters
e_h = 0                             # embedding dimension hyperparameters



### Test double integrator

## generative process

A = torch.tensor([[0, 1., 0], [0, 0, 0], [0, 0, 0]], device=DEVICE)             # state transition matrix
B_a = torch.tensor([[0, 0, 0], [0, 1., 0], [0, 0, 0]], device=DEVICE)           # input matrix (dynamics)
F = torch.tensor([[1., 0, 0], [0, 1., 0], [0, 0, 1.]], device=DEVICE)           # observation matrix

sigma_z_log = torch.tensor([12.])                                               # log-precision
sigma_z = torch.exp(sigma_z_log)
Sigma_z = torch.tensor([[sigma_z, 0., 0.], [0., sigma_z, 0.], 
            [0., 0, sigma_z]], device=DEVICE)                                   # TODO: tensor type deault is 'int64' when no dot is used, but need float for pseudo-inverse

sigma_w_log = torch.tensor([2.])                                                # log-precision
sigma_w = torch.exp(sigma_w_log)
Sigma_w = torch.tensor([[0., 0., 0.], [0., sigma_w, 0.], 
            [0., 0., 0.]], device=DEVICE)


## generative model
alpha = torch.exp(torch.tensor([2.]))
alpha2 = torch.exp(torch.tensor([1.]))
beta = torch.exp(torch.tensor([1.]))

A_gm = torch.tensor([[0, 1, 0], [-alpha, -alpha2, 0], [0, 0, 0]])               # state transition matrix
B_u_gm = torch.tensor([[0, 0, 0], [0, beta, 0], [0, 0, 0]])                     # input matrix (dynamics)
F_gm = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0]])                          # observation matrix

sigma_z_log_GM = torch.tensor([-1.])                                            # log-precision
sigma_z_GM = torch.exp(sigma_z_log)
Sigma_z_GM = torch.tensor([[sigma_z_GM, 0., 0.], [0., sigma_z_GM, 0.], 
            [0., 0, sigma_z_GM]], device=DEVICE)                                # TODO: tensor type deault is 'int64' when no dot is used, but need float for pseudo-inverse

sigma_w_log_GM = torch.tensor([-1.])                                            # log-precision
sigma_w_GM = torch.exp(sigma_w_log_GM)
Sigma_w_GM = torch.tensor([[sigma_w_GM, 0., 0.], [0., sigma_w_GM, 0.], 
            [0., 0, sigma_w_GM]], device=DEVICE)

sigma_v_log_GM = torch.tensor([-8.])                                            # log-precision
sigma_v_GM = torch.exp(sigma_v_log_GM)
Sigma_v_GM = torch.tensor([[sigma_v_GM, 0., 0.], [0., sigma_v_GM, 0.], 
            [0., 0, sigma_v_GM]], device=DEVICE)

dyda = torch.tensor([[1.], [1.], [1.]])
eta_u = torch.zeros(*dyda.shape)                                                 # desired state


GP = layer(T, dt, A=A, F=F, Sigma_w=Sigma_w_GM, Sigma_z=Sigma_z, e_n=e_n)
GM = layer(T, dt, A=A_gm, F=F_gm, Sigma_w=Sigma_w_GM, Sigma_z=Sigma_z_GM, e_n=e_n, dyda=dyda, B_u=B_u_gm, eta_u=eta_u)

learning_rate = 5e-3
learning_rate = 450000
learning_rate_action = 0.8

for i in range(iterations-1):
    print(i)

    GP.step(i)
    # GM.setObservations(GP.y.detach())
    GM.setObservations(GP.y)

    F = GM.free_energy(i)

    # Use autograd to compute the backward pass.
    F.backward()

    # Update weights using gradient descent
    dFdy = GM.eps_v                                                                         # dFdeps_v is in fact equal to dFdy since dFdy = dFdeps_v * deps_vdy with deps_vdy always equal to 1
    dFdx = GM.x.grad
    dFdu = GM.u.grad
    dFda = dFdy * GM.dyda
    with torch.no_grad():
        GM.x -= learning_rate * dt * dFdx
        GM.u -= learning_rate * dt * dFdu

        GP.a -= learning_rate_action * dt * dFda
    
        # Manually zero the gradients after updating weights
        GM.x.grad = None
        GM.u.grad = None
    
    GP.saveHistoryVariables(i)
    GM.saveHistoryVariables(i)

plt.figure()
plt.plot(GM.F_history[:-1].detach())

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(range(iterations-1), GM.y_history[:-1,0].detach(), 'b')
ax1.plot(range(iterations-1), GM.x_history[:-1,0].detach(), 'r')
ax1.plot(range(iterations-1), GP.x_history[:-1,0].detach(), 'k')

ax2.plot(range(iterations-1), GM.y_history[:-1,1].detach(), 'b')
ax2.plot(range(iterations-1), GM.x_history[:-1,1].detach(), 'r')
ax2.plot(range(iterations-1), GP.x_history[:-1,1].detach(), 'k')


fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
for i in range(GP.a_history.shape[1]):
    ax1.plot(range(iterations-1), GP.a_history[:-1,i].detach())


fig2 = plt.figure()
ax1 = fig2.add_subplot(131)
ax2 = fig2.add_subplot(132)
ax3 = fig2.add_subplot(133)

ax1.plot(range(iterations-1), GM.eps_v_history[:-1,0].detach(), 'b')
ax1.plot(range(iterations-1), GM.eps_v_history[:-1,1].detach(), 'r')
ax1.plot(range(iterations-1), GM.eps_v_history[:-1,2].detach(), 'k')

ax2.plot(range(iterations-1), GM.eps_x_history[:-1,0].detach(), 'b')
ax2.plot(range(iterations-1), GM.eps_x_history[:-1,1].detach(), 'r')
ax2.plot(range(iterations-1), GM.eps_x_history[:-1,2].detach(), 'k')

ax3.plot(range(iterations-1), GM.eps_eta_history[:-1,0].detach(), 'b')
ax3.plot(range(iterations-1), GM.eps_eta_history[:-1,1].detach(), 'r')
ax3.plot(range(iterations-1), GM.eps_eta_history[:-1,2].detach(), 'k')


fig3 = plt.figure()
ax1 = fig3.add_subplot(131)
ax2 = fig3.add_subplot(132)
ax3 = fig3.add_subplot(133)

ax1.plot(range(iterations-1), GM.xi_v_history[:-1,0].detach(), 'b')
ax1.plot(range(iterations-1), GM.xi_v_history[:-1,1].detach(), 'r')
ax1.plot(range(iterations-1), GM.xi_v_history[:-1,2].detach(), 'k')

ax2.plot(range(iterations-1), GM.xi_x_history[:-1,0].detach(), 'b')
ax2.plot(range(iterations-1), GM.xi_x_history[:-1,1].detach(), 'r')
ax2.plot(range(iterations-1), GM.xi_x_history[:-1,2].detach(), 'k')

ax3.plot(range(iterations-1), GM.xi_eta_history[:-1,0].detach(), 'b')
ax3.plot(range(iterations-1), GM.xi_eta_history[:-1,1].detach(), 'r')
ax3.plot(range(iterations-1), GM.xi_eta_history[:-1,2].detach(), 'k')


plt.show()