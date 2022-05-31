import sys
sys.path.append('src/modules')

import torch
import torch.nn as nn


from attention import MultiHeadAttention
from position_wise_ff import PositionWiseFF
from layer_normalization import LayerNormalization


class EncoderLayer(nn.Module):
    """Encoder Layer class"""
    def __init__(self, d_model, d_hidden, n_heads, dropout_prob):
        """Constructor for the Encoder Layer

        Args:
            d_model (int): shape of the of the input tensor or the last dimension of the input tensor
            n_hidden (int): dimensionality of the hidden layer of the FFNN
            n_heads (int): number of attention heads
            dropout_prob (float): dropout probability
        """
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.layer_norm1 = LayerNormalization(d_model=d_model)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.ffnn = PositionWiseFF(d_model=d_model, d_hidden=d_hidden, dropout_prob=dropout_prob)
        self.layer_norm2 = LayerNormalization(d_model=d_model)
        self.dropout2 = nn.Dropout(dropout_prob)


    def forward(self, x, mask=None):
        """Forward step of the EncoderLayer
        Attention ->+ Norm ->+ FF ->+ Norm

        Args:
            x (torch.Tensor): input tensor
            mask (torch.Tensor): optional mask for attention
        """

        # Save input tensor
        tmp = x
        # Compute self attention
        x = self.attention(query=x, key=x, value=x, mask=mask)

        # First Add and Normalize
        x = self.layer_norm1(tmp + x)
        x = self.dropout1(x)

        # PositionWide FF
        tmp = x
        x = self.ffnn(x)

        # Second Add and Normalize
        x = self.layer_norm2(tmp + x)
        x = self.dropout2(x)

        return x