# Reimplementation of "Connectionist Temporal Classification" (CTC) in PyTorch

This repository reimplements the model from the paper ["Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks"](https://dl.acm.org/doi/abs/10.1145/1143844.1143891) in PyTorch. The model is trained on the [LibriSpeech](http://www.openslr.org/12/) dataset for end-to-end speech recognition, using CTC loss to transcribe unsegmented audio into text.

## Requirements

- **Python** 3.8 or higher
- **PyTorch** 2.0.0 or higher
- **torchaudio** 2.0.0 or higher
- **tqdm** 4.60.0 or higher

To install dependencies, run:
```
pip install -r requirements.txt
```

## How to Use

### 1. Train the Model

To train the CTC-based model on the LibriSpeech dataset, run:

```
python main.py
```

This will:
- Download and preprocess the `train-clean-100` subset of LibriSpeech.
- Save the trained model to `model.pth`.
- Save the vocabulary to `char_to_idx.json`.

### 2. Test the Model

To evaluate the model on the unseen `test-clean` subset, run:

```
python test.py
```

This script will:
- Load `model.pth` and `char_to_idx.json`.
- Run inference and output decoded transcriptions for sample audio files.
