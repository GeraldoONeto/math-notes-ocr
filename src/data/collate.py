import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    images, tokens = zip(*batch)

    images = torch.stack(images)

    pad_tokens = pad_sequence(list(tokens), batch_first=True, padding_value=0)

    return images, pad_tokens