import torch
from torch import nn

a = torch.zeros([2, 3, 4], dtype=torch.float32)

b = torch.rand([1, 3, 4], dtype=torch.float32)

c = torch.cat([a, b], 0)
print(c)

d = torch.cat([a, b, c], 0)
print(d)