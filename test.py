import torch
from torch.utils.data import DataLoader
from model import CTCModel
from dataset import SpeechDataset, collate_fn, load_librispeech
from utils import greedy_decode, indices_to_text, load_vocab

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model(input_dim, hidden_dim, output_dim, model_path, num_layers=4, dropout=0.5):
    model = CTCModel(input_dim, hidden_dim, output_dim,
                     num_layers, dropout).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def test_model(model, dataloader, idx_to_char):
    model.eval()
    decoded_texts = []

    with torch.no_grad():
        for features, _, _, _ in dataloader:
            features = features.to(device)
            output = model(features).to("cpu")
            decoded_indices = greedy_decode(output)

            for seq in decoded_indices:
                decoded_texts.append(indices_to_text(seq, idx_to_char))

    return decoded_texts


def main():
    # Load vocabulary and create idx_to_char mapping
    char_to_idx = load_vocab("model/char_to_idx.json")
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    # Load test data
    features, labels = load_librispeech(
        "data/LibriSpeech", subset="test-clean", max_samples=10)
    dataset = SpeechDataset(features, labels, char_to_idx)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    # Initialize model
    input_dim = 64
    hidden_dim = 512
    output_dim = len(char_to_idx) - 1
    model = load_model(input_dim, hidden_dim, output_dim, "model/model.pth")

    # Test model and decode outputs
    decoded_texts = test_model(model, dataloader, idx_to_char)

    # Print some sample results
    print("Sample Test Results:")
    for i, text in enumerate(decoded_texts[:10]):
        print(f"Sample {i + 1}: {text}")


if __name__ == "__main__":
    main()
