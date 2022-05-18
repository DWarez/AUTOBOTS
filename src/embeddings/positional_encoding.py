from operator import index
from turtle import pos
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal encoding of the word embeddings"""
    def __init__(self, d_model, device, max_lenght=512):
        """Init the PositionalEncoding module

        Args:
            d_model (int): dimensionality of the words' embedding
            device (str): hardware device in use
            max_lenght (int, optional): maximum lenght of the sequence. Defaults to 512.
        """
        super(PositionalEncoding, self).__init__()

        # It's important that the condings have the same size of the embedding of the words
        # since the two must be summed
        self.pos_encoding = torch.zeros(max_lenght, d_model, device=device)
        self.pos_encoding.requires_grad = False

        # Column tensor for positions: [0, 1, 2, ..., max_lenght]^T
        positions = torch.arange(end=max_lenght, device=device).unsqueeze(dim=1)

        # Recall that:
        #   pe_i(w) = sin(w_i * t) if i%2 == 0
        #   pe_i(w) = cos(w_i * t) if i%2 != 0

        indexes = torch.arange(start=0, end=d_model, step=2, device=device)
        w_k = 10000.0 ** (indexes/d_model)
        
        # all odd dimensions
        self.pos_encoding[:, 0::2] = torch.sin(positions/w_k)
        # all even dimensions
        self.pos_encoding[:, 1::2] = torch.cos(positions/w_k)


    def forward(self, x):
        # return the sum of word embedding + positional embedding
        return x + self.pos_encoding[:, :]