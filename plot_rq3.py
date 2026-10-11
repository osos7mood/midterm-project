from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


Path("figures_rq3").mkdir(exist_ok=True)

summary = pd.read_csv("results_rq3/test_summary.csv")
summary = summary.set_index("model").loc[["cnn", "rnn"]]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    ["CNN", "RNN (LSTM)"],
    summary["test_accuracy"] * 100,
    color=["steelblue", "darkorange"]
)
ax.bar_label(bars, fmt="%.1f%%", padding=4)
ax.axhline(
    100 / 3, color="gray", linestyle="--",
    label="Chance (33.3%)"
)
ax.set_ylim(0, 110)
ax.set_ylabel("Test accuracy (%)")
ax.set_title("RQ3: Response-only classification\nHeld-out math/science domain")
ax.legend()
fig.tight_layout()
fig.savefig("figures_rq3/test_accuracy.png", dpi=300)
plt.close(fig)

for model, title in [("cnn", "CNN"), ("rnn", "RNN (LSTM)")]:
    matrix = pd.read_csv(
        f"results_rq3/{model}_output_confusion.csv",
        index_col=0
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        vmin=0,
        vmax=70,
        ax=ax
    )
    ax.set_xlabel("Predicted model")
    ax.set_ylabel("Actual model")
    ax.set_title(f"RQ3: {title}\nHeld-out math/science domain")
    fig.tight_layout()
    fig.savefig(
        f"figures_rq3/{model}_output_confusion.png",
        dpi=300
    )
    plt.close(fig)

print("Saved all 3 graphs in figures_rq3/")