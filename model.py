import os
import pandas as pd
import numpy as np
import torch
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# =========================
# CONFIG
# =========================
MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "final_complaints_dataset_with_categories.csv"
OUTPUT_DIR = "fast_model"
ENCODER_PATH = "label_encoder.pkl"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)
df = df.dropna()

df = df.sample(min(5000, len(df)), random_state=42)

texts = df["complaint"].astype(str).tolist()
labels = df["category"].astype(str).tolist()

# =========================
# LABEL ENCODING
# =========================
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
num_labels = len(label_encoder.classes_)

# =========================
# TRAIN / VAL SPLIT
# =========================
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_enc = tokenizer(train_texts, truncation=True, max_length=64)
val_enc = tokenizer(val_texts, truncation=True, max_length=64)

# =========================
# DATASET CLASS
# =========================
class ComplaintDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = ComplaintDataset(train_enc, train_labels)
val_dataset = ComplaintDataset(val_enc, val_labels)

# =========================
# MODEL
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

# =========================
# METRICS
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted"
    )

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# =========================
# TRAINING CONFIG (CPU OPTIMIZED)
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    evaluation_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,
    weight_decay=0.01,

    logging_steps=50,

    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0,

    save_total_limit=2,
    report_to="none"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =========================
# TRAINER
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# =========================
# TRAIN
# =========================
trainer.train()

# =========================
# EVALUATE
# =========================
results = trainer.evaluate()
print("Evaluation Results:", results)

# =========================
# SAVE MODEL + TOKENIZER
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# =========================
# SAVE LABEL ENCODER
# =========================
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ MODEL TRAINING COMPLETE")
print(f"📦 Saved to: {OUTPUT_DIR}")