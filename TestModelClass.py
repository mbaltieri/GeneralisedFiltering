from layer import *
import functions
import matplotlib.pyplot as plt

# torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(precision=10)


# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device in use:", DEVICE)

torch.manual_seed(42)

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

A = torch.tensor([[0, 1., 0], [0, 0, 0], [0, 0, 0]], device=DEVICE)                 # state transition matrix
B_a = torch.tensor([[0, 0, 0], [0, 1., 0], [0, 0, 0]], device=DEVICE)               # input matrix (dynamics)
F = torch.tensor([[1., 0, 0], [0, 1., 0], [0, 0, 1.]], device=DEVICE)               # observation matrix

sigma_z_log = torch.tensor([14.], device=DEVICE)                                     # log-precision
sigma_z = torch.exp(sigma_z_log)
Sigma_z = torch.tensor([[sigma_z, 0., 0.], [0., sigma_z, 0.], 
            [0., 0, sigma_z]], device=DEVICE)                                       # TODO: tensor type deault is 'int64' when no dot is used, but need float for pseudo-inverse

sigma_w_log = torch.tensor([-6.], device=DEVICE)                                     # log-precision
sigma_w = torch.exp(sigma_w_log)
Sigma_w = torch.tensor([[0., 0., 0.], [0., sigma_w, 0.], 
            [0., 0., 0.]], device=DEVICE)


## generative model
alpha = torch.exp(torch.tensor([2.]))
alpha2 = torch.exp(torch.tensor([1.]))
beta = torch.exp(torch.tensor([0.]))

A_gm = torch.tensor([[0, 1, 0], [-alpha, -alpha2, 0], [0, 0, 0]], device=DEVICE)    # state transition matrix
# A_gm = torch.zeros(3,3, device=DEVICE)    # state transition matrix
B_u_gm = torch.tensor([[beta, 0, 0], [0, beta, 0], [0, 0, beta]], device=DEVICE)    # input matrix (dynamics)
F_gm = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0]], device=DEVICE)               # observation matrix

sigma_z_log_GM = torch.tensor([-2.], device=DEVICE)                                  # log-precision
sigma_z_GM = torch.exp(sigma_z_log_GM)
Sigma_z_GM = torch.tensor([[sigma_z_GM, 0., 0.], [0., sigma_z_GM, 0.], 
            [0., 0, sigma_z_GM]], device=DEVICE)                                    # TODO: tensor type deault is 'int64' when no dot is used, but need float for pseudo-inverse

sigma_w_log_GM = torch.tensor([11.], device=DEVICE)                                 # log-precision
sigma_w_GM = torch.exp(sigma_w_log_GM)
Sigma_w_GM = torch.tensor([[sigma_w_GM, 0., 0.], [0., sigma_w_GM, 0.], 
            [0., 0, sigma_w_GM]], device=DEVICE)

sigma_v_log_GM = torch.tensor([15.], device=DEVICE)                                 # log-precision
sigma_v_GM = torch.exp(sigma_v_log_GM)
Sigma_v_GM = torch.tensor([[sigma_v_GM, 0., 0.], [0., sigma_v_GM, 0.], 
            [0., 0, sigma_v_GM]], device=DEVICE)

dyda = 1*torch.tensor([[0., 10., 0.], [0., 1., 0.], [0., 1., 0.]], device=DEVICE)
eta_u = torch.tensor([[0.], [0.], [0.]], device=DEVICE)                            # desired state


## create models
GP = layer(T, dt, A=A, F=F, Sigma_w=Sigma_w_GM, Sigma_z=Sigma_z, e_n=e_n, B_a=B_a)
GM = layer(T, dt, A=A_gm, F=F_gm, Sigma_w=Sigma_w_GM, Sigma_z=Sigma_z_GM, Sigma_v=Sigma_v_GM, e_n=e_n, dyda=dyda, B_u=B_u_gm, eta_u=eta_u)

learning_rate = 5e-3
learning_rate = 1
learning_rate_action = 1000.

for i in range(iterations-1):
    GP.saveHistoryVariables(i)
    GM.saveHistoryVariables(i)

    print(i)

    GP.step(i)
    GM.setObservations(GP.y)

    # calculate loss
    F = GM.free_energy(i)

    # Use autograd to compute the backward pass
    F.backward()

    # Update weights using gradient descent
    dFdy = GM.y.grad[i,:]
    dFdx = GM.x.grad[i,:]
    dFdu = GM.u.grad[i,:]
    dFda = GM.dyda @ dFdy
    with torch.no_grad():
        GM.x[i+1,:] = GM.x[i,:] - learning_rate * dt * dFdx
        GM.u[i+1,:] = GM.u[i,:] - learning_rate * dt * dFdu

        # GP.a[i+1,:] = GP.a[i,:] - learning_rate_action * dt * dFda
    
        # Manually zero the gradients after updating weights
        GM.y.grad = None
        GM.x.grad = None
        GM.u.grad = None
    
    # print(GP.x_history[:i,1])
    # print(GP.a_history[:i,1])


plt.figure()
plt.plot(GM.y_history[:-1,0].detach(), GM.y_history[:-1,1].detach(), 'b')
plt.plot(GM.x_history[:-1,0].detach(), GM.x_history[:-1,1].detach(), 'r')

# fig6 = plt.figure()
# ax1 = fig6.add_subplot(121)
# ax1.plot(range(iterations-1), GM.x_history[:-1,0].detach(), 'r')


fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

# ax1.plot(range(iterations-1), GP.y_history[:-1,0].detach(), 'y')
ax1.plot(range(iterations-1), GM.y_history[:-1,0].detach(), 'b')
ax1.plot(range(iterations-1), GM.x_history[:-1,0].detach(), 'r')
ax1.plot(range(iterations-1), GP.x_history[:-1,0].detach(), 'k')
ax1.plot(range(iterations-1), GM.u_history[:-1,0].detach(), 'g')

# ax2.plot(range(iterations-1), GP.y_history[:-1,1].detach(), 'y')
ax2.plot(range(iterations-1), GM.y_history[:-1,1].detach(), 'b')
ax2.plot(range(iterations-1), GM.x_history[:-1,1].detach(), 'r')
ax2.plot(range(iterations-1), GP.x_history[:-1,1].detach(), 'k')
ax2.plot(range(iterations-1), GM.u_history[:-1,1].detach(), 'g')

# ax3.plot(range(iterations-1), GP.y_history[:-1,2].detach(), 'y')
# ax3.plot(range(iterations-1), GM.y_history[:-1,2].detach(), 'b')
ax3.plot(range(iterations-1), GM.x_history[:-1,2].detach(), 'r')
# ax3.plot(range(iterations-1), GP.x_history[:-1,2].detach(), 'k')
ax3.plot(range(iterations-1), GM.u_history[:-1,2].detach(), 'g')


fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
# ax1.plot(range(iterations-1), GP.a_history[:-1,0].detach(), 'b')
ax1.plot(range(iterations-1), GP.a_history[:-1,1].detach(), 'r')
# ax1.plot(range(iterations-1), GP.a_history[:-1,2].detach(), 'k')
# for i in range(GP.a_history.shape[1]):
#     ax1.plot(range(iterations-1), GP.a_history[:-1,i].detach())


fig2 = plt.figure(figsize=(15,5))
plt.title('Prediction errors')
ax1 = fig2.add_subplot(131)
ax2 = fig2.add_subplot(132)
ax3 = fig2.add_subplot(133)

ax1.plot(range(iterations-1), GM.eps_v_history[:-1,0].detach(), 'b')
ax1.plot(range(iterations-1), GM.eps_v_history[:-1,1].detach(), 'r')
ax1.plot(range(iterations-1), GM.eps_v_history[:-1,2].detach(), 'k')

# ax2.plot(range(iterations-1), GM.eps_x_history[:-1,0].detach(), 'b')
# ax2.plot(range(30,iterations-1), GM.eps_x_history[30:-1,1].detach(), 'r')
# ax2.plot(range(30,iterations-1), GM.eps_x_history[30:-1,2].detach(), 'k')

ax2.plot(range(iterations-1), GM.eps_x_history[:-1,0].detach(), 'b')
ax2.plot(range(iterations-1), GM.eps_x_history[:-1,1].detach(), 'r')
ax2.plot(range(iterations-1), GM.eps_x_history[:-1,2].detach(), 'k')

ax3.plot(range(iterations-1), GM.eps_eta_history[:-1,0].detach(), 'b')
ax3.plot(range(iterations-1), GM.eps_eta_history[:-1,1].detach(), 'r')
ax3.plot(range(iterations-1), GM.eps_eta_history[:-1,2].detach(), 'k')


fig3 = plt.figure(figsize=(15,5))
plt.title('Weighted prediction errors')

# ax4 = fig3.add_subplot(141)
ax1 = fig3.add_subplot(131)
ax2 = fig3.add_subplot(132)
ax3 = fig3.add_subplot(133)

# ax4.plot(range(iterations-1), dFdy_h[:-1,0].detach(), 'b')
# ax4.plot(range(iterations-1), dFdy_h[:-1,1].detach(), 'r')
# ax4.plot(range(iterations-1), dFdy_h[:-1,2].detach(), 'k')

ax1.plot(range(iterations-1), GM.xi_v_history[:-1,0].detach(), 'b')
ax1.plot(range(iterations-1), GM.xi_v_history[:-1,1].detach(), 'r')
ax1.plot(range(iterations-1), GM.xi_v_history[:-1,2].detach(), 'k')

# ax2.plot(range(iterations-1), GM.xi_x_history[:-1,0].detach(), 'b')
# ax2.plot(range(30,iterations-1), GM.xi_x_history[30:-1,1].detach(), 'r')
# ax2.plot(range(30,iterations-1), GM.xi_x_history[30:-1,2].detach(), 'k')

ax2.plot(range(iterations-1), GM.xi_x_history[:-1,0].detach(), 'b')
ax2.plot(range(iterations-1), GM.xi_x_history[:-1,1].detach(), 'r')
ax2.plot(range(iterations-1), GM.xi_x_history[:-1,2].detach(), 'k')

ax3.plot(range(iterations-1), GM.xi_eta_history[:-1,0].detach(), 'b')
ax3.plot(range(iterations-1), GM.xi_eta_history[:-1,1].detach(), 'r')
ax3.plot(range(iterations-1), GM.xi_eta_history[:-1,2].detach(), 'k')


plt.show()