import json
from pathlib import Path

import torch


def encode_label(label, char_to_idx):
    return torch.tensor([char_to_idx[char] for char in label])


def greedy_decode(output, blank=0):
    pred = output.argmax(dim=2)
    decoded_sequences = []
    for sequence in pred:
        decoded = []
        prev_char = None
        for idx in sequence:
            if idx != blank and idx != prev_char:
                decoded.append(idx.item())
            prev_char = idx
        decoded_sequences.append(decoded)
    return decoded_sequences


def indices_to_text(indices, idx_to_char):
    return ''.join([idx_to_char[i] for i in indices if i != 0])


def save_vocab(char_to_idx, filepath="model/char_to_idx.json"):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(char_to_idx, f)


def load_vocab(filepath="model/char_to_idx.json"):
    with open(filepath, "r") as f:
        return json.load(f)
