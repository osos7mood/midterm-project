import torch
from torch import nn


class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes=3):
        super().__init__()

        # Represent each token with 100 learned numbers.
        self.embedding = nn.Embedding(
            vocab_size, 100, padding_idx=0
        )

        # Look for patterns spanning 3, 4, or 5 tokens.
        self.convs = nn.ModuleList([
            nn.Conv1d(100, 64, kernel_size=k)
            for k in (3, 4, 5)
        ])

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(64 * 3, num_classes)

    def forward(self, x):
        # x contains token IDs, with padding at the end.
        lengths = (x != 0).sum(dim=1)
        embedded = self.embedding(x).transpose(1, 2)

        pooled = []

        for conv in self.convs:
            features = torch.relu(conv(embedded))

            # Ignore windows containing only padding.
            positions = torch.arange(
                features.size(2), device=x.device
            )
            valid = positions[None, :] < lengths[:, None]
            features = features.masked_fill(
                ~valid[:, None, :], float("-inf")
            )

            pooled.append(features.amax(dim=2))

        combined = torch.cat(pooled, dim=1)
        return self.classifier(self.dropout(combined))

class TextRNN(nn.Module):
    def __init__(self, vocab_size, num_classes=3):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, 100, padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=100,
            hidden_size=64,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        lengths = (x != 0).sum(dim=1).cpu()
        embedded = self.embedding(x)

        # Skip padding so it does not affect the final state.
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        return self.classifier(self.dropout(hidden[-1]))

if __name__ == "__main__":
    example = torch.tensor([
        [2, 3, 4, 5, 6, 0],
        [7, 8, 9, 0, 0, 0]
    ])

    cnn = TextCNN(vocab_size=100)
    rnn = TextRNN(vocab_size=100)

    print("CNN output shape:", cnn(example).shape)
    print("RNN output shape:", rnn(example).shape)