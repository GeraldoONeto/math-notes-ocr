import os
import re
import glob
import torch
from PIL import Image

from torch.utils.data import Dataset
"""
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
        tensor_image = self.transform(image)


        tokens = self.tokenizer.encode(text)
        final_tokens = [1] + tokens + [2]
        tensor_tokens = torch.tensor(final_tokens)


        return tensor_image, tensor_tokens

class MathWritingDataset(Dataset):
    def __init__(self, dataset, transform, tokenizer):
        self.dataset = dataset
        self.transform = transform 
        self.tokenizer = tokenizer

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        image = self.dataset[index]['image']
        text = self.dataset[index]['latex']

        tensor_images = self.transform(image)

        tokens = self.tokenizer.encode(text)
        tokens = [1] + tokens + [2]
        tensor_tokens = torch.tensor(tokens)

        return tensor_images, tensor_tokens
"""

class HMEDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
        self.im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        ex = self.dataset[index]
        image = ex["image"].convert("RGB")

        saida = self.processor(image, return_tensors="pt", format=True)
        prompt_ids = saida["inputs_ids"][0]

        alvo_ids = self.processor.tokenizer(ex["text"])["input_ids"] + [self.im_end_id]
        alvo_ids = torch.tensor(alvo_ids)

        input_ids = torch.cat([prompt_ids, alvo_ids])
        labels = torch.cat([torch.full_like(prompt_ids, -100), alvo_ids])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": saida["pixel_values"][0]
        }