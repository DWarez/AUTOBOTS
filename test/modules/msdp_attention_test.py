import sys
sys.path.append('src/modules')

import torch
import torch.nn as nn

from attention import MultiHeadAttention

query = torch.zeros(1, 100, 20)
key = torch.zeros(1, 100, 20)
value = torch.zeros(1, 100, 20)

module = MultiHeadAttention(query.shape[-1], 4)

print(module.forward(query, key, value).shape)