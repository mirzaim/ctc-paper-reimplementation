import torch
import torch.nn as nn
from torch.utils.data import Dataset

import torchaudio
from torchaudio.datasets import LIBRISPEECH

from utils import encode_label


def load_librispeech(root_dir, subset="train-clean-100", max_samples=1000, n_mels=64):
    dataset = LIBRISPEECH(root=root_dir, url=subset, download=True)
    features, labels = [], []

    for idx, (waveform, _, transcript, _, _, _) in enumerate(dataset):
        if idx >= max_samples:
            break
        spectrogram = torchaudio.transforms.MelSpectrogram(
            n_mels=n_mels)(waveform)
        log_spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)
        features.append(log_spectrogram.squeeze(0).T)  # Shape: (time, freq)
        labels.append(transcript)

    return features, labels


class SpeechDataset(Dataset):
    def __init__(self, features, labels, char_to_idx):
        self.features = features
        self.labels = [encode_label(label, char_to_idx) for label in labels]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def collate_fn(batch):
    features, labels = zip(*batch)
    feature_lengths = torch.tensor([f.shape[0]
                                   for f in features], dtype=torch.long)
    label_lengths = torch.tensor([len(label)
                                 for label in labels], dtype=torch.long)
    features = nn.utils.rnn.pad_sequence(features, batch_first=True)
    labels = torch.cat(labels)
    return features, labels, feature_lengths, label_lengths
