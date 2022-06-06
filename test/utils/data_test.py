import sys
sys.path.append('src/utils')
sys.path.append('src/embeddings')

from data import WikiText2Dataset

dataset = WikiText2Dataset(128, 35, "cpu")

train, val, test = dataset.get_datasets()
print(f"Train set shape: {train.shape}")

print(f"Vocalubary length: {len(dataset.vocab)}")

print(train[0][:])
