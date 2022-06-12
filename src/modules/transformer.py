import sys
sys.path.append('src/modules')
sys.path.append('src/embeddings')

import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, v_size_enc: int, v_size_dec: int, 
                    seq_length: int, d_model: int, n_heads: int, d_hidden: int,
                    n_layers: int, src_pad_idx: int, trg_pad_idx: int,
                    dropout_prob: float, device: str) -> None:
        """_summary_

        Args:
            v_size_enc (int): size of the encoder vocabulary
            v_size_dec (int): size of the decoder vocabulary 
                                (output of fc linear layer)
            seq_length (int): (max) length of the sequence
            d_model (int): token embedding size
            n_heads (int): number of attention heads
            d_hidden (int): size of the hidden layers
            n_layers (int): number of layers for encoding/decoding
            src_pad_idx (int): identifier of the <pad> token for source
            trg_pad_idx (int): identifier of the <pad> token for target
            dropout_prob (float): dropout probability
            device (str): device to use
        """
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        self.encoder = Encoder(v_size=v_size_enc, 
                                seq_length=seq_length, d_model=d_model,
                                d_hidden=d_hidden, n_heads=n_heads,
                                n_layers=n_layers, dropout_prob=dropout_prob,
                                device=device)

        self.decoder = Decoder(v_size_dec=v_size_dec,
                                seq_length=seq_length, d_model=d_model,
                                d_hidden=d_hidden, n_heads=n_heads,
                                n_layers=n_layers, dropout_prob=dropout_prob,
                                device=device)


    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Generates the mask for the source

        Args:
            src (torch.Tensor): source with shape
                [batch_size, src_length]

        Returns:
            torch.Tensor: source mask with shape [batch_size, 1, 1, src_lenght]
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask


    def make_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        """Generates the mask for the target (no peak mask) by combining the 
        pad mask with the sub mask (lower triangular matrix).

        Args:
            trg (torch.Tensor): target tensor with shape
                [batch_size, trg_lenght]

        Returns:
            torch.Tensor: target mask tensor with shape
                [batch_size, 1, trg_lenght, trg_length]
        """
        trg_len = trg.shape[1]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len),
                                                device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask


    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """Forward step of the transformer
        Generate source mask -> generate target mask -> pass source and source
        mask into encoder -> collect output of decoder using target+target mask
        and source+source mask.

        Args:
            src (torch.Tensor): source tensor with shape
                [batch_size, src_legth]
            trg (torch.Tensor): target tensor with shape
                [batch_size, trg_length]

        Returns:
            torch.Tensor: output of the decoder block with shape
                [batch_size, target_length, vocabulary_size_decoder]
        """
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc = self.encoder(src, src_mask)
        output = self.decoder(trg, enc, trg_mask, src_mask)

        return output