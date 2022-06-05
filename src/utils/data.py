import torch
from torch.utils.data import dataset

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import os

import warnings
warnings.filterwarnings("ignore")

def create_data_directory() -> str:
    """Creates the directory to store the dataset

    Returns:
        str: path where the dataset is stored
    """
    try:
        path = str(os.getcwd()) + "\data"
        os.mkdir(path=path)
    except OSError:
        print(f"Couldn't make directory {path}")
        return os.path.expanduser('~/.torchtext/cache')
    else:
        print(f"Created directory {path}")
        return path


class WikiText2Dataset():
    """WikiText2Dataset Class"""
    def __init__(self, batch_size: int, bptt: int, device: str) -> None:
        """WikiText2Dataset constructor

        Args:
            batch_size (int): batch size
            bptt (int): chuncks length
            device (str): device to use
        """
        self.batch_size = batch_size
        self.bptt = bptt
        self.device = device
        self.path = create_data_directory()

        self.train_iter = WikiText2(root=self.path, split='train')
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, \
                                        self.train_iter), specials=['<unk>'])
        self.vocab.set_default_index(self.vocab['<unk>'])

        train_i, val_i, test_i = WikiText2()
        self.train_iter = self.data_process(train_i)
        self.val_iter = self.data_process(val_i)
        self.test_iter = self.data_process(test_i)


    def data_process(self, \
                    raw_text_iter: dataset.IterableDataset) -> torch.Tensor:
        """Converts raw text into a flat Tensor. Black Magic"""
        data = [torch.tensor(self.vocab(self.tokenizer(item)), \
                                dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


    def batchify(self, data: torch.Tensor) -> torch.Tensor:
        """Creates batches of data from the flat Tensor

        Args:
            data (torch.Tensor): flat Tensor from self.data_process()

        Returns:
            torch.Tensor: batched tensor
        """
        sequence_length = data.size(0)//self.batch_size
        data = data[:sequence_length * self.batch_size]
        data = data.view(self.batch_size, sequence_length).contiguous()
        return data.to(self.device)


    def get_datasets(self) -> tuple[torch.Tensor]:
        """Returns train, validation and test sets

        Returns:
            tuple[torch.Tensor]: batchified train/val/test set
        """
        return self.batchify(self.train_iter), self.batchify(self.val_iter), \
                                                self.batchify(self.test_iter)


    def get_batch(self, src: torch.Tensor, index: int):
        sequence_length = min(self.bptt, src.shape[-1] - index - 1)
        data = src[index:index+sequence_length]
        target = src[index+1:index+1+sequence_length].reshape(-1)
        return data, target