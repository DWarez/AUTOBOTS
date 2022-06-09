import sys
sys.path.append('src/modules')
sys.path.append('src/embeddings')

import torch
import torch.nn as nn

from attention import MultiHeadAttention
from layer_normalization import LayerNormalization
from position_wise_ff import PositionWiseFF
from transformer_embedding import TransformerEmbedding

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, n_heads: int, 
                                            dropout_prob: float) -> None:
        """Decoder Layer constructor

        Args:
            d_model (int): size of token embedding
            d_hidden (int): size of hidden layers
            n_heads (int): number of attention heads
            dropout_prob (float): dropout probability
        """
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.layer_norm = LayerNormalization(d_model=d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.ffnn = PositionWiseFF(d_model=d_model, d_hidden=d_hidden)

    
    def forward(self, dec: torch.Tensor, enc: torch.Tensor, 
                                    target_mask: torch.Tensor, 
                                    source_mask: torch.Tensor) -> torch.Tensor:
        """Forward step
        x -> self attention -> layer normalization -> dropout
        (repeat if enc is not none) ->
        -> ffnn -> layer normalization -> dropout

        Args:
            dec (torch.Tensor): decoder input
            enc (torch.Tensor): output of encoder
            target_mask (torch.Tensor): mask for target tensor
            source_mask (torch.Tensor): mask for encoder output

        Returns:
            torch.Tensor: output of decoding
        """
        x = self.attention(query=dec, key=dec, value=dec, 
                                                        mask=target_mask)
        x = self.layer_norm(x + dec)
        x = self.dropout(x)

        if enc is not None:
            _x = x
            x = self.attention(query=x, key=enc, value=enc, mask=source_mask)
            x = self.layer_norm(x + _x)
            x = self.dropout(x)
        
        _x = x
        x = self.ffnn(x)
        x = self.layer_norm(x + _x)
        x = self.dropout(x)
        
        return x


class Decoder(nn.Module):
    def __init__(self, v_size_dec: int, batch_size: int, seq_length: int, 
                    d_model: int, d_hidden: int, n_heads: int, 
                    n_layers: int, dropout_prob: float, device: str) -> None:
        """Decoder constructor

        Args:
            v_size_dec (int): size of the decoder vocabulary
            batch_size (int): batch size 
            seq_length (int): length of the sequence
            d_model (int): size of token embedding
            d_hidden (int): size of hidden layers
            n_heads (int): number of attention heads
            n_layers (int): number of layers
            dropout_prob (float): dropout probability
            device (str): device to use
        """
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(batch_size=batch_size, 
                                                v_size=v_size_dec,
                                                seq_length=seq_length,
                                                d_model=d_model,
                                                dropout_prob=dropout_prob,
                                                device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                    d_hidden=d_hidden,
                                                    n_heads=n_heads,
                                                    dropout_prob=dropout_prob)
                                    for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, v_size_dec)


    def forward(self, target: torch.Tensor, source: torch.Tensor, 
                                    target_mask: torch.Tensor, 
                                    source_mask: torch.Tensor) -> torch.Tensor:
        """Decoder forward step
            x -> embedding -> decoding layers -> linear -> output

        Args:
            target (torch.Tensor): target tensor
            source (torch.Tensor): encoder output tensor
            target_mask (torch.Tensor): target mask tensor
            source_mask (torch.Tensor): encoder output mask tensor

        Returns:
            torch.Tensor: _description_
        """
        x = self.embedding(target.long())

        for layer in self.layers:
            x = layer(x, source, target_mask, source_mask)
        
        x = self.linear(x)
        return x