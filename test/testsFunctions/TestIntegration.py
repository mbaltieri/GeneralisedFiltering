import numpy as np
import matplotlib.pyplot as plt

import torch
import math
from functions import Diff

T = 10
dt = 0.01
iterations = int(T//dt)

# x = np.zeros((iterations, 1))

# for i in range(iterations-1):
#     dx = np.sin(i*2*np.pi/100)
#     x[i+1] = x[i] + dt*dx

# plt.figure()
# plt.plot(x)


sim = 3
e_sim = 0

x = torch.zeros(iterations, sim*(e_sim+1), 1, requires_grad = True)
A = torch.tensor([[0, 1., 0], [0., 0., 0], [0, 0, 0]])

x[0,:] = torch.normal(0, 1., size=(sim*(e_sim+1), 1))

def f(i, A, x):
    # return - 3 * x[i,:]
    # return torch.sin(torch.tensor([i*2*math.pi/100]))
    return A @ x[i,:]# + self.B_u @ self.u[i,:] + self.B_a @ self.a[i,:]

for i in range(iterations-1):
    dx = f(i, A, x)# + self.C @ self.w.noise[i,:].unsqueeze(1)
    x[i+1,:] = x[i,:] + dt * dx
    # self.y[i,:] = self.g(i) + self.H @ self.z.noise[i,:].unsqueeze(1)


fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.plot(range(iterations-1), x[:-1,0].detach(), 'k')
ax2.plot(range(iterations-1), x[:-1,1].detach(), 'k')
ax3.plot(range(iterations-1), x[:-1,2].detach(), 'k')


plt.show()