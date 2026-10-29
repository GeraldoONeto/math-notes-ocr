import torch
from torch.utils.data import Dataset

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

        proc_out = self.processor(image, return_tensors="pt", format=True)
        prompt_ids = proc_out["input_ids"][0]

        target_ids = self.processor.tokenizer(ex["text"])["input_ids"] + [self.im_end_id]
        target_ids = torch.tensor(target_ids)

        input_ids = torch.cat([prompt_ids, target_ids])
        labels = torch.cat([torch.full_like(prompt_ids, -100), target_ids])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": proc_out["pixel_values"][0]
        }