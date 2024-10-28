import torch.nn as nn


class CTCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(CTCModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(2 * hidden_dim, output_dim + 1)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        logits = self.fc(rnn_out)
        return logits.log_softmax(2)
