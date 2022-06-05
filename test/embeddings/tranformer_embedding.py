import sys
sys.path.append('src/utils')
sys.path.append('src/embeddings')

from data import WikiText2Dataset
from transformer_embedding import TransformerEmbedding

dataset = WikiText2Dataset(128, 35, "cpu")

train, val, test = dataset.get_datasets()

embed = TransformerEmbedding(batch_size=128, v_size=len(dataset.vocab), 
                                d_model=20, seq_length=train.shape[-1], 
                                dropout_prob=0.1, device="cpu")

print(embed(train).shape)