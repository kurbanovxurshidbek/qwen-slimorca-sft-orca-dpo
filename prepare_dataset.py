from datasets import load_dataset

# ============================================================
# Load English DPO (preference) dataset
# ============================================================
# Intel/orca_dpo_pairs columns: system, question, chosen, rejected
#   chosen   = GPT-4 answer         (preferred)
#   rejected = Llama-2-13B-chat answer (not preferred)

dataset = load_dataset(
    "Intel/orca_dpo_pairs",
    split="train"
)

df = dataset.to_pandas()
print(f"Loaded {len(df)} preference pairs")

# ============================================================
# Clean
# ============================================================

for col in ["system", "question", "chosen", "rejected"]:
    df[col] = df[col].fillna("").astype(str).str.strip()

# drop pairs with an empty question / answer
df = df[(df["question"] != "") & (df["chosen"] != "") & (df["rejected"] != "")]

# drop pairs where chosen == rejected (no preference signal)
df = df[df["chosen"] != df["rejected"]]

# drop exact duplicates
df = df.drop_duplicates(subset=["system", "question", "chosen", "rejected"])

print(f"After cleaning: {len(df)} pairs")

# ============================================================
# Convert to DPO messages format
# ============================================================
# messages = prompt (system + user), chosen / rejected = one assistant reply.
# Empty system prompt is left out, so the Qwen chat template adds its default
# system prompt — exactly like in SFT and in lm-evaluation-harness.

messages_col, chosen_col, rejected_col = [], [], []

for _, row in df.iterrows():

    messages = []

    if row["system"]:
        messages.append({"role": "system", "content": row["system"]})

    messages.append({"role": "user", "content": row["question"]})

    messages_col.append(messages)
    chosen_col.append({"role": "assistant", "content": row["chosen"]})
    rejected_col.append({"role": "assistant", "content": row["rejected"]})

df["messages"] = messages_col
df["chosen"] = chosen_col
df["rejected"] = rejected_col

# ============================================================
# Save JSONL
# ============================================================

df = df.sample(frac=1.0, random_state=42)   # shuffle (seed 42, same as SFT)

df[["messages", "chosen", "rejected"]].to_json(
    "train.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)

print(f"Saved {len(df)} preference pairs")
print("Output: train.jsonl")
