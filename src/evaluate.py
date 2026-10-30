import re
import torch


from rapidfuzz.distance import Levenshtein

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def predict(model, processor, image):
    inputs = processor(image, return_tensors='pt').to(DEVICE)
    out = model.generate(
        **inputs,
        do_sample=False,
        tokenizer=processor.tokenizer,
        stop_strings="<|im_end|>",
        max_new_tokens=256
    )
    novos = out[0, inputs["input_ids"].shape[1]:]

    return processor.decode(novos, skip_special_tokens=True).strip()

def normalize(latex):
    s = latex.strip()
    s = re.sub(r"^\\\[|\\\]$", "", s)
    s = re.sub(r"^\$+|\$+$", "", s)
    s = re.sub(r"\s+", "", s)
    return s

def score(pred, ans):
    p, t = normalize(pred), normalize(ans)
    dist = Levenshtein.distance(p, t)
    
    return dist, dist / max(len(t), 1)