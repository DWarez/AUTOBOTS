import sys
sys.path.append('src/embeddings')

import torch
import torch.nn as nn

from positional_encoding import PositionalEncoding


device = "cpu"

_tensor = torch.randn(10, 100, 20)

pos_encoding = PositionalEncoding(_tensor.shape[0], _tensor.shape[-1], device, max_lenght=100)

result = pos_encoding.forward(_tensor)

print(result.shape)