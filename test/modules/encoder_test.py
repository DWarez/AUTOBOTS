import sys
sys.path.append('src/modules')
sys.path.append('src/utils')
sys.path.append('src/embeddings')

import torch
from encoder import Encoder


_tensor = torch.zeros([32, 128])

embed = Encoder(v_size=2000, seq_length=128, d_model=20,
                    d_hidden=5, n_heads=4, n_layers=1,
                    dropout_prob=0.1, device="cpu")

print(embed(_tensor).shape)