import torch
from torch.nn.utils.rnn import pad_sequence

def make_collate_fn(pad_token_id):
    def collate_fn(batch):
        input_ids = pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_token_id
        )
        labels = pad_sequence(
            [b["labels"] for b in batch], batch_first=True, padding_value=-100
        )
        attention_mask = pad_sequence(
            [torch.ones_like(b["input_ids"]) for b in batch], batch_first=True, padding_value=0
        )
        pixel_values = torch.stack([b["pixel_values"] for b in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values
        }
    
    return collate_fn