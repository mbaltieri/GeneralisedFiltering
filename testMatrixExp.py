import numpy as np
import torch
from scipy.linalg import expm


A = torch.tensor([[20.0880156754, 20.0855369232,  7.3890560989],
        [20.0855369232, 20.0880156754,  7.3890560989],
        [ 7.3890560989,  7.3890560989,  2.7182818285]])

B = np.array([[20.0880156754, 20.0855369232,  7.3890560989],
        [20.0855369232, 20.0880156754,  7.3890560989],
        [ 7.3890560989,  7.3890560989,  2.7182818285]])

C = torch.matrix_exp(A)
D = expm(B)

print(C)
print(D)



# they seem to give the same results