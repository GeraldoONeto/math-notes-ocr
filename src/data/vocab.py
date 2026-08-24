import os
import re
import json
import glob

TOKEN_PATTERN = r'\\[a-zA-Z]+|.'

def build_vocab(dataset_folder):
    
    txt_paths = glob.glob(os.path.join(dataset_folder, '*.txt'))

    tokens = set()

    for path in txt_paths:
        with open(path, encoding='utf-8')  as f:
            text = f.read()
        tokens.update(re.findall(TOKEN_PATTERN, text))

    sort_tokens = sorted(tokens)
    
    vocab = {
        '<PAD>': 0,
        '<BOS>': 1,
        '<EOS>': 2
        }
    vocab.update({token: id for id, token in enumerate(sort_tokens, start=3)})

    return vocab


def invert_vocab(vocab):
    return {value: key for key, value in vocab.items()}


def save_vocab(vocab, json_path):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(json_path):
    with open(json_path, encoding='utf-8') as f:
        return json.load(f)