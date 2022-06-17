import sys
sys.path.append('src/modules')

import torch
import torch.nn as nn

from position_wise_ff import PositionWiseFF


_tensor = torch.ones([1, 512, 20])

layer = PositionWiseFF(20, 100)
result = layer.forward(_tensor)

assert result.shape == (1, 512, 20), f"Incorrect embedding shape, expected (1, 512, 20), got {result.shape}"