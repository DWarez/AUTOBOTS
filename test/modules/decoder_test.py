import sys
sys.path.append('src/modules')
sys.path.append('src/utils')
sys.path.append('src/embeddings')

import torch

from decoder import Decoder
from encoder import Encoder



_tensor = torch.ones([32, 128])

encoder = Encoder(v_size=2000, seq_length=128, d_model=20,
                                d_hidden=5, n_heads=4, n_layers=1,
                                dropout_prob=0.1, device="cpu")

decoder = Decoder(v_size_dec=2000, seq_length=128, d_model=20,
                    d_hidden=10, n_heads=4, n_layers=1, dropout_prob=0.1, 
                    device="cpu")

src = encoder(_tensor)

print(decoder(_tensor, src, None, None).shape)