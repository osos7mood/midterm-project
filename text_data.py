import re
from collections import Counter

import pandas as pd


def tokenize(text):
    # Separate words and punctuation.
    return re.findall(r"\w+|[^\w\s]", str(text).lower())


def build_vocab(texts):
    counts = Counter()

    for text in texts:
        counts.update(tokenize(text))

    # PAD fills empty positions; UNK represents unfamiliar words.
    vocab = {"<PAD>": 0, "<UNK>": 1}

    for word, count in counts.most_common(20000):
        if count >= 2:
            vocab[word] = len(vocab)

    return vocab

import torch
from torch.utils.data import Dataset, DataLoader

LABELS = {"GPT": 0, "Claude": 1, "Gemini": 2}


class TextDataset(Dataset):
    def __init__(self, df, vocab, mode="output", max_length=512):
        columns = {
            "input": df["LLM_Input"],
            "output": df["LLM_output"],
            "both": df["LLM_Input"] + " " + df["LLM_output"]
        }

        self.texts = columns[mode].tolist()
        self.labels = df["LLM_name"].map(LABELS).tolist()
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        tokens = tokenize(self.texts[index])[:self.max_length]

        # Use UNK if a text has no tokens.
        ids = [self.vocab.get(token, 1) for token in tokens] or [1]

        # Pad every sequence to the same length.
        ids += [0] * (self.max_length - len(ids))

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long)
        )

if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    vocab = build_vocab(train["LLM_output"])

    example = tokenize("Hello, how are you?")
    numbers = [vocab.get(word, 1) for word in example]

    print("Vocabulary size:", len(vocab))
    print("Example tokens:", example)
    print("Example numbers:", numbers)
    if __name__ == "__main__":
        dataset = TextDataset(train, vocab, mode="output")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    texts, labels = next(iter(loader))

    print("Text batch shape:", texts.shape)
    print("Label batch shape:", labels.shape)