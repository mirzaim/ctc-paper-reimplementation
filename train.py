from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import CTCModel
from dataset import load_librispeech, SpeechDataset, collate_fn
from utils import save_vocab

from tqdm import tqdm


def train(model, dataloader, optimizer, scheduler, device, num_epochs=20):
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as tepoch:
            for features, labels, input_lengths, target_lengths in tepoch:
                features, labels = features.to(device), labels.to(device)
                input_lengths, target_lengths = input_lengths.to(
                    device), target_lengths.to(device)

                optimizer.zero_grad()
                output = model(features)
                loss = ctc_loss(output.transpose(0, 1), labels,
                                input_lengths, target_lengths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_epoch_loss}")
        scheduler.step(avg_epoch_loss)

    return model


def main():
    # Set device
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load data
    Path('data/LibriSpeech').mkdir(parents=True, exist_ok=True)
    features, labels = load_librispeech("data/LibriSpeech", max_samples=10000)
    vocab = sorted(set(c for label in labels for c in label))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(vocab)}
    char_to_idx["blank"] = 0
    save_vocab(char_to_idx)

    dataset = SpeechDataset(features, labels, char_to_idx)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    # Initialize model, optimizer, and scheduler
    input_dim = 64
    hidden_dim = 512
    output_dim = len(char_to_idx) - 1
    model = CTCModel(input_dim, hidden_dim, output_dim,
                     num_layers=4, dropout=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2)

    # Train the model
    model = train(model, dataloader, optimizer,
                  scheduler, device, num_epochs=20)

    # Save model and optimizer
    torch.save(model.state_dict(), "model/model.pth")
    torch.save(optimizer.state_dict(), "model/optimizer.pth")
    torch.save(scheduler.state_dict(), "model/scheduler.pth")


if __name__ == "__main__":
    main()
