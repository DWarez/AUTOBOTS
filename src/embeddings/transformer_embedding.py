import torch
import torch.nn as nn

from positional_encoding import PositionalEncoding


class TokenEmbedding(nn.Embedding):
    """Embedding of tokens"""
    def __init__(self, v_size: int, d_model: int) -> None:
        """Constructor of TokenEmbedding

        Args:
            v_size (int): size of the vocabulary
            d_model (int): size of embedding
        """
        super(TokenEmbedding, self).__init__(v_size, d_model, padding_idx=1)


class TransformerEmbedding(nn.Module):
    """Embedding of tokens for the Transformer"""
    def __init__(self, v_size: int, seq_length: int, d_model: int, 
                                    dropout_prob: float, device: str) -> None:
        """Embeds the words into TokenEmbeddings and applies positional 
        encoding

        Args:
            v_size (int): vocabulary size
            d_model (int): size of embedding
            max_length (int): max length of sequence
            dropout_prob (float): dropout probability
            device (str): device to use
        """
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(v_size=v_size, d_model=d_model)
        self.positional_encoding = PositionalEncoding(seq_length=seq_length, 
                                                d_model=d_model, device=device)
        self.dropout = nn.Dropout(dropout_prob)

    
    def forward(self, x):
        """tensor -> token embedding + positional encoding -> dropout
        Args:
            x (torch.Tensor): tokens to embed 
        """
        te = self.token_embedding(x.long())
        pe = self.positional_encoding(te)
        return self.dropout(pe)