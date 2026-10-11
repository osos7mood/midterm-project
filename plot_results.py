from pathlib import Path
from matplotlib.container import BarContainer
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv("results/test_summary.csv")

table = results.pivot(
    index="mode",
    columns="model",
    values="test_accuracy"
)

table = table.reindex(["input", "output", "both"])
table = table[["cnn", "rnn"]] * 100
table.index = pd.Index(
    ["Prompt only", "Response only", "Prompt + response"]
)
table.columns = ["CNN", "RNN (LSTM)"]

ax = table.plot.bar(
    figsize=(9, 5),
    color=["#2563eb", "#f59e0b"],
    rot=0
)

for bars in ax.containers:
    if isinstance(bars, BarContainer):
        ax.bar_label(bars, fmt="%.1f%%", padding=3)

ax.axhline(
    100 / 3,
    color="gray",
    linestyle="--",
    label="Chance level (33.3%)"
)

ax.set_title("LLM Identification: Test Accuracy")
ax.set_xlabel("")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 110)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

plt.tight_layout()

Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/test_accuracy.png", dpi=300)
plt.close()

print("Saved figures/test_accuracy.png")