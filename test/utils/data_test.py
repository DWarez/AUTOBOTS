import sys
sys.path.append('src/utils')
sys.path.append('src/embeddings')

from data import WikiText2Dataset
from positional_encoding import PositionalEncoding
from torch.nn import Embedding

dataset = WikiText2Dataset(128, 35, "cpu")

train, val, test = dataset.get_datasets()
print(f"Train set shape: {train.shape}")

data, target = dataset.get_batch(train, 10)

print(f"Data shape: {data.shape}")
print(f"Target shape: {target.shape}")

embed = Embedding(len(dataset.vocab), 20)
e_data = embed(data)
print(f"Embedded data shape: {e_data.shape}")

pe = PositionalEncoding(35, e_data.shape[1], 20, device="cpu")
pe_data = pe(e_data)
print(f"PE data shape: {pe_data.shape}")
