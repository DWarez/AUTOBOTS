import sys
sys.path.append('src/modules')
sys.path.append('src/embeddings')

import torch
import torch.nn as nn


from attention import MultiHeadAttention
from position_wise_ff import PositionWiseFF
from layer_normalization import LayerNormalization
from transformer_embedding import TransformerEmbedding


class EncoderLayer(nn.Module):
    """Encoder Layer class"""
    def __init__(self, d_model: int, d_hidden: int, n_heads: int, 
                                                        dropout_prob: float):
        """Constructor for the Encoder Layer

        Args:
            d_model (int): shape of the of the input tensor or the last
                dimension of the input tensor
            n_hidden (int): dimensionality of the hidden layer of the FFNN
            n_heads (int): number of attention heads
            dropout_prob (float): dropout probability
        """
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, 
                                                    n_heads=n_heads)
        self.layer_norm1 = LayerNormalization(d_model=d_model)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.ffnn = PositionWiseFF(d_model=d_model, d_hidden=d_hidden, 
                                                    dropout_prob=dropout_prob)
        self.layer_norm2 = LayerNormalization(d_model=d_model)
        self.dropout2 = nn.Dropout(dropout_prob)


    def forward(self, x: torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
        """Forward step of the EncoderLayer
        Attention ->+ Norm ->+ FF ->+ Norm

        Args:
            x (torch.Tensor): input tensor
            mask (torch.Tensor): optional mask for attention
        
        Returns:
            x (torch.Tensor): output of computation
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


class Encoder(nn.Module):
    """Encoder module"""
    def __init__(self, v_size: int, seq_length: int, 
                    d_model: int, d_hidden: int, n_heads: int, n_layers: int, 
                                            dropout_prob: float, device: str):
        """Encoder module of Transformer architecture

        Args:
            v_size (int): vocabulary size
            seq_length (int): length of the sequence
            d_model (int): size of token embedding
            d_hidden (int): size of hidden layers
            n_heads (int): number of attention heads
            n_layers (int): number of EncoderLayers that will compose the
                Encoder block
            dropout_prob (float): dropout probability
            device (str): device to use
        """
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(v_size=v_size, d_model=d_model, 
                                                seq_length=seq_length, 
                                                dropout_prob=dropout_prob, 
                                                device=device)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, 
                                                    d_hidden=d_hidden, 
                                                    n_heads=n_heads, 
                                                    dropout_prob=dropout_prob) 
                                        for _ in range(n_layers)])

    
    def forward(self, x: torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
        """Forward step of Encoder block

        Args:
            x (torch.Tensor): input tensor
            mask (torch.Tensor, optional): mask to apply. Defaults to None.

            x -> embedding -> Encoder layers -> output

        Returns:
            torch.Tensor: result of computation
        """
        x = self.embedding(x.long())
        for layer in self.layers:
            x = layer(x, mask=mask)

        return x