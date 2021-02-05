import torch

a = torch.arange(0,15)

print(a)

b = a[x for i,x in enumerate(a) if i!=3]