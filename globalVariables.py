import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_large = torch.exp(torch.tensor(32.))
_small = torch.exp(torch.tensor(-32.))