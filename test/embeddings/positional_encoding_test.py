import sys
sys.path.append('src/embeddings')

import torch
import torch.nn as nn

from positional_encoding import PositionalEncoding


device = "cpu"

pos_encoding = PositionalEncoding(20, device, max_lenght=100)

result = pos_encoding.forward(torch.zeros(1, 100, 20))

print(result.shape)
print(result[0][0][:])