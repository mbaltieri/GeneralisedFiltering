from Model import *

dt = 1
T = 1
iterations = int(T/dt)

l = 3                               # number of layers

m = 2                               # observations dimension
n = 2                               # hidden states dimension
r = 2                               # inputs dimension
p = 6                               # parameters dimension
h = 3                               # hyperparameters dimension

e_n = 3                             # embedding dimension hidden states
e_r = 0                             # embedding dimension inputs
e_p = 0                             # embedding dimension parameters
e_h = 0                             # embedding dimension hyperparameters

delta = 1.50                                                # parameters simulating a simple mass-spring system as the environment
epsilon = 15.0

A = torch.tensor([[0, 1], [delta, epsilon]])
F = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0]])

Sigma_w = torch.tensor([[3., 0.], [0., 1.]])
Sigma_z = torch.tensor([[2., 0., 0., 0.], [0., 1., 0., 0.], [0., 0, .5, 0.], [0., 0., 0., .25]])                       # TODO: tensor type deault is 'int64' but need float for pseudo-inverse

GP = layer(A=A+10, F=F, Sigma_w=Sigma_w, Sigma_z=Sigma_z, e_n=e_n)
GM = layer(A=A, F=F, Sigma_w=Sigma_w, Sigma_z=Sigma_z, e_n=e_n)

for i in range(iterations):
    GP.forward()
    GM.updateObservations(GP.y)

    GM.backward()

    GM.forward(flag = 0)

    
