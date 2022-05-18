import sys
sys.path.append('src/models')

import torch
import torch.nn as nn

from layer_normalization import LayerNormalization


_tensor = torch.zeros(1, 100, 20)

layer_norm = LayerNormalization(_tensor.shape)

result = layer_norm.forward(_tensor)

print(result.shape)