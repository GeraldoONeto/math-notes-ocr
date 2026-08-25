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
        self.file_names = []
        self.texts = []

        png_paths = glob.glob(os.path.join(self.folder, '*.png'))
        
        for path in png_paths:
            name = os.path.basename(path).replace('.png', '')
            self.file_names.append(name)

            txt_path = os.path.join(self.folder, name + '.txt')    
            with open(txt_path, encoding='utf-8') as f:
                self.texts.append(f.read())

    def __len__(self):
        return len(self.file_names)


    def __getitem__(self, index):
        name = self.file_names[index]
        text = self.texts[index]

        png_path = os.path.join(self.folder, name + '.png')
        image = Image.open(png_path)
        image = self.transform(image)


        tokens = self.tokenizer.encode(text)
        final_tokens = [1] + tokens + [2]
        tensor_tokens = torch.tensor(final_tokens)


        return image, tensor_tokens