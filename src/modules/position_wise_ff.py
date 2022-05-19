import torch
import torch.nn as nn

from collections import OrderedDict

class PositionWiseFF(nn.Module):
    """Position Wise Feed Forward Module"""

    def __init__(self, d_model, d_hidden, dropout_prob=0.1):
        """PositionWiseFF constructor

        Args:
            d_model (int): shape of the of the input tensor or the last dimension of the input tensor
            d_hidden (int): shape of the hidden representation
            dropout_prob (float, optional): dropout probability. Defaults to 0.1.
        """
        super(PositionWiseFF, self).__init__()
        self.d_model = d_model if isinstance(d_model, int) else d_model[-1]
        self.d_hidden = d_hidden
        self.dropout_prob = dropout_prob

        self.net = nn.Sequential(OrderedDict({
            "linear1": nn.Linear(self.d_model, self.d_hidden),
            "relu": nn.ReLU(),
            "dropout": nn.Dropout(self.dropout_prob),
            "linear2": nn.Linear(self.d_hidden, self.d_model)
        }))

    
    def forward(self, x):
        return self.net(x)