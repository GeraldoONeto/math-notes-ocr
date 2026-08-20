import os
import re
import glob
import torch
from PIL import Image

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from . import vocab

class CROHMEDataset(Dataset):
    def __init__(self, folder, transform, vocabulary):
        self.folder = folder
        self.transform = transform
        self.vocabulary = vocabulary 

        png_paths = glob.glob(os.path.join(self.folder, '*.png'))
        self.file_names = []
        
        for path in png_paths:
            name = os.path.basename(path).replace('.png', '')

            self.file_names.append(name)


    def __len__(self):
        return len(self.file_names)


    def __getitem__(self, index):
        name = self.file_names[index]

        png_path = os.path.join(self.folder, name + '.png')
        txt_path = os.path.join(self.folder, name + '.txt')

        image = Image.open(png_path).convert('RGB')
        image = self.transform(image)

        with open(txt_path, encoding='utf-8') as f:
            text = f.read()

        token_keys = re.findall(vocab.TOKEN_PATTERN, text)
        tokens_values = []

        for key in token_keys:
            tokens_values.append(self.vocabulary[key])

        tensor_tokens = torch.tensor(tokens_values)

        return image, tensor_tokens


def collate_fn(batch):
    images, tokens = zip(*batch)

    images = torch.stack(images)

    pad_tokens = pad_sequence(list(tokens), batch_first=True, padding_value=0)

    return images, pad_tokens