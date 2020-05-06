from Model import *

l = 3                               # number of layers

m = 2                               # observations dimension
n = 2                               # hidden states dimension
r = 2                               # inputs dimension
p = 6                               # parameters dimension
h = 3                               # hyperparameters dimension

e_m = 4                             # embedding dimension observations
e_n = 4                             # embedding dimension hidden states
e_r = 4                             # embedding dimension inputs
e_p = 1                             # embedding dimension parameters
e_h = 1                             # embedding dimension hyperparameters

delta = 1.50                                                # parameters simulating a simple mass-spring system as the environment
epsilon = 15.0

A = torch.tensor([[0, 1], [delta, epsilon]])
F = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0]])

# model = layer(A=A, F=F, e_n=e_n, e_m=e_m)

B=torch.zeros(1)
print(len(B))

B=torch.empty(1)
print(len(B))

print(model.B)
print(model.v)
b = model.B @ model.v

a = model.f()