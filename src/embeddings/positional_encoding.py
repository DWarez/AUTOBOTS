import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal encoding of the word embeddings"""
    def __init__(self, batch_size: int, d_model: int, device: str, max_lenght=5000, dropout_prob=0.1) -> None:
        """Init the PositionalEncoding module

        Args:
            batch_size (int): size of the batch.
            d_model (int): dimensionality of the words' embedding.
            device (str): hardware device in use.
            max_lenght (int, optional): maximum lenght of the sequence. Defaults to 5000.
            dropout_prob (int, optional): probability of applying dropout. Defaults to 0.1.
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout_prob)

        # It's important that the condings have the same size of the embedding of the words
        # since the two must be summed
        self.pos_encoding = torch.zeros(batch_size, max_lenght, d_model, device=device)
        self.pos_encoding.requires_grad = False

        # Column tensor for positions: [0, 1, 2, ..., max_lenght]^T
        positions = torch.arange(start=0, end=max_lenght, device=device).unsqueeze(dim=1)

        # Recall that:
        #   pe_i(w) = sin(w_i * t) if i%2 == 0
        #   pe_i(w) = cos(w_i * t) if i%2 != 0

        indexes = torch.arange(start=0, end=d_model, step=2, device=device)
        divider = torch.exp(indexes * (-math.log(10000.0)/d_model))

        # all odd dimensions
        self.pos_encoding[:, :, 0::2] = torch.sin(positions * divider)
        # all even dimensions
        self.pos_encoding[:, :, 1::2] = torch.cos(positions * divider)


    def forward(self, x: torch.Tensor):
        """Positional enconding forward step

        Args:
            x (torch.Tensor): input tensor. Shape [batch_size, sequence_length, d_model]

        Returns:
            torch.Tensor: positional encoding
        """
        # return the sum of word embedding + positional embedding
        x = x + self.pos_encoding[:, :x.size(1), :]
        return self.dropout(x)