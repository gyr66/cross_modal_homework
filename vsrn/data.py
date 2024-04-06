import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import nltk


def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, idx = zip(*data)
    images = torch.stack(images, dim=0)
    valid_length = torch.tensor([len(caption) for caption in captions])
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    mask = captions != 0
    return images, captions, mask, valid_length, idx


class F30kDataset(Dataset):
    def __init__(
        self,
        data_path,
        data_split,
        vocab,
    ):
        self.vocab = vocab
        self.images = np.load(os.path.join(data_path, "%s_ims.npy" % data_split))
        self.captions = []
        with open(os.path.join(data_path, "%s_caps.txt" % data_split), "r") as f:
            for line in f:
                self.captions.append(line.strip())
        self.length = len(self.captions)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])
        caption = self.captions[idx]

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab("<start>"))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab("<end>"))
        caption = torch.tensor(caption)

        return image, caption, idx

    def __len__(self):
        return self.length
