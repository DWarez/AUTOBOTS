import sys
sys.path.append('src/modules')

import torch
import torch.nn as nn

from position_wise_ff import PositionWiseFF


_tensor = torch.ones([1, 512, 20])

layer = PositionWiseFF(20, 100)
print(layer.forward(_tensor).shape)

layer = PositionWiseFF(_tensor.shape, 100)
print(layer.forward(_tensor).shape)