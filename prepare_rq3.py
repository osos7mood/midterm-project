from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel("LLM_Responses_Domains.xlsx")

# Keep math/science entirely separate for testing.
test = df[df["domain"] == "math_science"].copy()
source = df[df["domain"] != "math_science"].copy()

# Each prompt must belong to exactly one topic.
assert df.groupby("prompt_id")["domain"].nunique().eq(1).all()

# Split the remaining prompts into training and validation.
prompts = source[["prompt_id", "domain"]].drop_duplicates()

train_ids, val_ids = train_test_split(
    prompts["prompt_id"],
    test_size=0.20,
    random_state=42,
    stratify=prompts["domain"],
)

train = source[source["prompt_id"].isin(train_ids)]
val = source[source["prompt_id"].isin(val_ids)]

assert set(train["prompt_id"]).isdisjoint(val["prompt_id"])
assert set(train["prompt_id"]).isdisjoint(test["prompt_id"])
assert set(val["prompt_id"]).isdisjoint(test["prompt_id"])

folder = Path("data_rq3")
folder.mkdir(exist_ok=True)

for name, split in [("train", train), ("val", val), ("test", test)]:
    split.to_csv(folder / f"{name}.csv", index=False)
    print(f"\n{name}: {len(split)} responses")
    print(split["LLM_name"].value_counts())

print("\nSaved RQ3 splits in data_rq3.")