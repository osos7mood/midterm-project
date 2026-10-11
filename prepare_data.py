import pandas as pd

df = pd.read_excel("LLM_Responses (2).xlsx")

print("Total responses:", len(df))
print("\nResponses per model:")
print(df["LLM_name"].value_counts())

print("\nMissing values:")
print(df.isna().sum())
from sklearn.model_selection import train_test_split
from pathlib import Path

prompt_ids = df["prompt_id"].unique()

train_ids, remaining_ids = train_test_split(
    prompt_ids, test_size=0.30, random_state=42
)

val_ids, test_ids = train_test_split(
    remaining_ids, test_size=0.50, random_state=42
)

train = df[df["prompt_id"].isin(train_ids)]
val = df[df["prompt_id"].isin(val_ids)]
test = df[df["prompt_id"].isin(test_ids)]

Path("data").mkdir(exist_ok=True)

train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("\nTraining rows:", len(train))
print("Validation rows:", len(val))
print("Testing rows:", len(test))