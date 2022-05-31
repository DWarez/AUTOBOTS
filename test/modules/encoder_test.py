import sys
sys.path.append('src/modules')

import torch
import torch.nn as nn

from encoder import EncoderLayer


_tensor = torch.zeros(1, 100, 20)

encoder_layer = EncoderLayer(_tensor.shape[-1], 10, 4, 1e-10)

print(encoder_layer)