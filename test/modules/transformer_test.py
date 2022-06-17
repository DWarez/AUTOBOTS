import sys
sys.path.append('src/modules')
sys.path.append('src/utils')
sys.path.append('src/embeddings')

import torch

from transformer import Transformer

_tensor = torch.ones([32, 128])

model = Transformer(v_size_enc=200, v_size_dec=200,
                    seq_length=128, d_model=20, n_heads=4, d_hidden=12,
                    n_layers=2, src_pad_idx=2, trg_pad_idx=2, dropout_prob=0.1,
                    device="cpu")

result = model(_tensor, _tensor)

assert result.shape == (32, 128, 2000), f"Incorrect embedding shape, expected (32, 128, 2000), got {result.shape}"