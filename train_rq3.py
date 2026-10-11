import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from models import TextCNN, TextRNN
from text_data import TextDataset, build_vocab, LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "rnn"], default="cnn")
    parser.add_argument(
        "--mode", choices=["input", "output", "both"], default="output"
    )
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)

    train = pd.read_csv("data_rq3/train.csv")
    val = pd.read_csv("data_rq3/val.csv")

    # Build the vocabulary from the selected training text only.
    if args.mode == "input":
        texts = train["LLM_Input"]
    elif args.mode == "output":
        texts = train["LLM_output"]
    else:
        texts = train["LLM_Input"] + " " + train["LLM_output"]

    vocab = build_vocab(texts)

    train_loader = DataLoader(
        TextDataset(train, vocab, mode=args.mode),
        batch_size=16,
        shuffle=True
    )
    val_loader = DataLoader(
        TextDataset(val, vocab, mode=args.mode),
        batch_size=16,
        shuffle=False
    )

    model_class = TextCNN if args.model == "cnn" else TextRNN
    model = model_class(vocab_size=len(vocab))

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    Path("checkpoints_rq3").mkdir(exist_ok=True)
    Path("results_rq3").mkdir(exist_ok=True)

    name = f"{args.model}_{args.mode}"
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, 11):
        model.train()
        train_loss = 0
        train_correct = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            scores = model(inputs)
            loss = loss_function(scores, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(labels)
            train_correct += (scores.argmax(1) == labels).sum().item()

        model.eval()
        val_loss = 0
        val_correct = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                scores = model(inputs)
                loss = loss_function(scores, labels)

                val_loss += loss.item() * len(labels)
                val_correct += (scores.argmax(1) == labels).sum().item()

        train_loss /= len(train)
        val_loss /= len(val)
        train_accuracy = train_correct / len(train)
        val_accuracy = val_correct / len(val)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy
        })

        print(
            f"Epoch {epoch:02d}/10 | "
            f"Train accuracy: {train_accuracy:.1%} | "
            f"Val accuracy: {val_accuracy:.1%} | "
            f"Val loss: {val_loss:.4f}",
            flush=True
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save({
                "model_state": model.state_dict(),
                "vocab": vocab,
                "labels": LABELS,
                "model_type": args.model,
                "mode": args.mode,
                "max_length": 512,
                "epoch": epoch
            }, f"checkpoints_rq3/{name}.pt")

    pd.DataFrame(history).to_csv(
        f"results_rq3/{name}_history.csv", index=False
    )
    print(f"\nSaved best model: checkpoints_rq3/{name}.pt")


if __name__ == "__main__":
    main()