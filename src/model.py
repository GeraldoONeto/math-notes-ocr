import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"

def load_model(device, freeze_encoder=True, dtype=torch.float32, checkpt_path = None):
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=dtype, device_map=device
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    if freeze_encoder:
        for p in model.model.vision_tower.parameters():
            p.requires_grad = False

    if checkpt_path is not None:
        state = torch.load(checkpt_path, map_location=device)
        result = model.load_state_dict(state, strict=False)
        print("unexpected keys:", result.unexpected_keys)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6:.1f}M")
    return model, processor