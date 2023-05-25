from attention import *
import torch
from torch import nn

inputs = torch.rand((2, 16, 64, 64))
print('inputs shape: ', inputs.shape)

se = SEAttention(16)
print('-------------SE----------------')
print(se)
se_outputs = se(inputs)
print(se_outputs)

eca = ECAAttention(16)
print('-------------ECA----------------')
print(eca)
eca_outputs = eca(inputs)
print(eca_outputs)

cbam = CBAMAttention(16)
print('-------------CBAM----------------')
print(cbam)
cbam_outputs = cbam(inputs)
print(cbam_outputs)

ca = CAAttention(16)
print('-------------CA----------------')
print(ca)
ca_outputs = cbam(inputs)
print(ca_outputs)

