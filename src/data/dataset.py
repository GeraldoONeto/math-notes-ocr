import os
import re
import glob
import torch
from PIL import Image

from torch.utils.data import Dataset
from pix2tex.cli import LatexOCR

class CROHMEDataset(Dataset):
    def __init__(self, folder, transform, tokenizer):
        self.folder = folder
        self.transform = transform
        self.tokenizer = tokenizer 

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

        tokens = self.tokenizer.encode(text)

        final_tokens = [1] + tokens + [2]

        tensor_tokens = torch.tensor(final_tokens)


        return image, tensor_tokens