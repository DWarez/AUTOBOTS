import torch
import torch.nn as nn

from positional_encoding import PositionalEncoding


class TokenEmbedding(nn.Embedding):
    """Embedding of tokens"""
    def __init__(self, v_size, d_model):
        """Constructor of TokenEmbedding

        Args:
            v_size (int): size of the vocabulary
            d_model (int): last dimension of the input tensor
        """
        super(TokenEmbedding, self).__init__(v_size, d_model, padding_idx=1)


class TransformerEmbedding(nn.Module):
    """Embedding of tokens for the Transformer"""
    def __init__(self, v_size, d_model, max_length, dropout_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(v_size=v_size, d_model=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, device=device, max_lenght=max_length)
        self.dropout = nn.Dropout(dropout_prob)

    
    def forward(self, x):
        """tensor -> token embedding + positional encoding -> dropout
        Args:
            x (torch.Tensor): tokens to embed 
        """
        te = self.token_embedding(x)
        pe = self.positional_encoding(x)
        return self.dropout(te + pe)