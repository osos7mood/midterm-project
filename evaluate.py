from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from models import TextCNN, TextRNN
from text_data import TextDataset, LABELS


def main():
    torch.set_num_threads(4)

    # Check that all six trained models exist before testing.
    names = [
        f"{model}_{mode}"
        for model in ("cnn", "rnn")
        for mode in ("input", "output", "both")
    ]

    missing = [
        name for name in names
        if not Path(f"checkpoints/{name}.pt").exists()
    ]

    if missing:
        print("Train these models first:", ", ".join(missing))
        return

    test = pd.read_csv("data/test.csv")
    Path("results").mkdir(exist_ok=True)

    class_names = list(LABELS)
    class_ids = list(LABELS.values())
    summary = []

    for name in names:
        checkpoint = torch.load(
            f"checkpoints/{name}.pt",
            map_location="cpu",
            weights_only=True
        )

        model_class = (
            TextCNN if checkpoint["model_type"] == "cnn"
            else TextRNN
        )
        model = model_class(len(checkpoint["vocab"]))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        dataset = TextDataset(
            test,
            checkpoint["vocab"],
            mode=checkpoint["mode"],
            max_length=checkpoint["max_length"]
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        actual = []
        predicted = []

        with torch.no_grad():
            for inputs, labels in loader:
                scores = model(inputs)
                predicted.extend(scores.argmax(1).tolist())
                actual.extend(labels.tolist())

        accuracy = accuracy_score(actual, predicted)
        macro_f1 = f1_score(
            actual, predicted,
            labels=class_ids,
            average="macro",
            zero_division=0
        )

        summary.append({
            "model": checkpoint["model_type"],
            "mode": checkpoint["mode"],
            "saved_epoch": checkpoint["epoch"],
            "test_accuracy": accuracy,
            "macro_f1": macro_f1
        })

        report = classification_report(
            actual, predicted,
            labels=class_ids,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        pd.DataFrame(report).T.to_csv(
            f"results/{name}_report.csv"
        )

        matrix = confusion_matrix(
            actual, predicted, labels=class_ids
        )
        pd.DataFrame(
            matrix,
            index=class_names,
            columns=class_names
        ).to_csv(f"results/{name}_confusion.csv")

        predictions = test[["prompt_id", "LLM_name"]].copy()
        id_to_name = {value: key for key, value in LABELS.items()}
        predictions["predicted_model"] = [
            id_to_name[value] for value in predicted
        ]
        predictions.to_csv(
            f"results/{name}_predictions.csv", index=False
        )

        print(
            f"{name}: accuracy={accuracy:.1%}, "
            f"macro F1={macro_f1:.3f}"
        )

    pd.DataFrame(summary).to_csv(
        "results/test_summary.csv", index=False
    )
    print("\nSaved results/test_summary.csv")


if __name__ == "__main__":
    main()