import sys
sys.path.append('src/embeddings')

import torch
import torch.nn as nn

from positional_encoding import PositionalEncoding

device = "cpu"

_tensor = torch.randn(2, 10, 6)

pos_encoding = PositionalEncoding(_tensor.shape[0], _tensor.shape[-1], device)

result = pos_encoding.forward(_tensor)

print(result.shape)
print(result)