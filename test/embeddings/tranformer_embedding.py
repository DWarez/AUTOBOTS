import sys
sys.path.append('src/utils')
sys.path.append('src/embeddings')

import torch
from transformer_embedding import TransformerEmbedding

_tensor = torch.ones(128, 35)

embed = TransformerEmbedding(v_size=200, d_model=20, 
                                seq_length=35, dropout_prob=0.1, 
                                device="cpu")

result = embed(_tensor).shape

assert result.shape == (128, 35, 20), f"Incorrect embedding shape, expected (128, 35, 20), got {result.shape}"