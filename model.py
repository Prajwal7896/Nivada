import pandas as pd
import torch
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

df = pd.read_csv("final_complaints_dataset_with_categories.csv")
df = df.sample(10000, random_state=42)

texts = df["complaint"].astype(str).tolist()
labels = df["category"].astype(str).tolist()

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
num_labels = len(set(labels))

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=64
)

val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=64
)

class ComplaintDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ComplaintDataset(train_encodings, train_labels)
val_dataset = ComplaintDataset(val_encodings, val_labels)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ==============================
# ⚙️ TRAINING ARGUMENTS
# ==============================
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_steps=100,
    fp16=torch.cuda.is_available(),  
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# ==============================
# 📊 EVALUATION
# ==============================
results = trainer.evaluate()
print("📊 FINAL RESULTS:")
print(results)

# ==============================
# 💾 SAVE MODEL
# ==============================
model.save_pretrained("fast_model")
tokenizer.save_pretrained("fast_model")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Model Saved Successfully!")

# ==============================
# 🚀 LOAD ONCE FOR FAST PREDICTION
# ==============================
tokenizer = DistilBertTokenizer.from_pretrained("fast_model")
model = DistilBertForSequenceClassification.from_pretrained("fast_model")
model.eval()

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def predict_complaint(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = torch.argmax(outputs.logits).item()
    return label_encoder.inverse_transform([pred_id])[0]

print(predict_complaint("No electricity in my area for 3 days"))