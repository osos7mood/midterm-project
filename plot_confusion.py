from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Path("figures").mkdir(exist_ok=True)

for model in ("cnn", "rnn"):
    for mode in ("input", "output", "both"):
        name = f"{model}_{mode}"

        matrix = pd.read_csv(
            f"results/{name}_confusion.csv",
            index_col=0
        )

        fig, ax = plt.subplots(figsize=(6, 5))

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            vmin=0,
            vmax=42,
            ax=ax
        )

        ax.set_title(f"{model.upper()} — {mode}")
        ax.set_xlabel("Predicted LLM")
        ax.set_ylabel("Actual LLM")

        fig.tight_layout()
        fig.savefig(
            f"figures/{name}_confusion.png",
            dpi=300
        )
        plt.close(fig)

print("Saved all six confusion matrices in figures.")