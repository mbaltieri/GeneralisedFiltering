from Model import *

l = 3                               # number of layers

m = 2                               # observations dimension
n = 2                               # hidden states dimension
r = 2                               # inputs dimension
p = 6                               # parameters dimension
h = 3                               # hyperparameters dimension

e_n = 4                             # embedding dimension hidden states
e_r = 4                             # embedding dimension inputs
e_p = 1                             # embedding dimension parameters
e_h = 1                             # embedding dimension hyperparameters

delta = 1.50                                                # parameters simulating a simple mass-spring system as the environment
epsilon = 15.0

A = torch.tensor([[0, 1], [delta, epsilon]])
F = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0]])

C = torch.tensor([[3., 0.], [0., 1.]])
H = torch.tensor([[2., 0.], [0., 1.], [.5, 0.], [.25, 1.]])                       # TODO: tensor type deault is 'int64' but need float for pseudo-inverse

model = layer(A=A, F=F, C=C, H=H, e_n=e_n)

b = model.A @ model.x
# print(b)
# print(Diff(b.double(), 2, e_n+1))

model.prediction_errors()

free = model.free_energy()

# print(model.F.size())
# print(model.x.size())

# print(model.G.size())
# print(model.v.size())

# print(model.H.size())
# print(model.z.size())

# print(model.C.size())
# print(model.w.size())


# print(model.n)
# print(model.w)

a = model.f()
c = model.g()

# print(model.H)
# print(model.C)