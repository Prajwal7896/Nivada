# AI-Powered Smart Complaint Management System

## Overview

This project is an AI-based Complaint Management System that automatically classifies public complaints, assigns them to the correct department, and provides basic intelligent solutions using Retrieval-Augmented Generation (RAG).

The system was built using Flask, DistilBERT, SQLite, and Sentence Transformers.

The main goal of the project is to reduce manual complaint handling and improve response efficiency in government and public service departments.

---

# Features

- User Authentication System
- Admin Department Login
- AI-Based Complaint Classification
- Automatic Department Routing
- Basic RAG-Based Solution Generation
- Complaint Tracking Dashboard
- Image Upload Support
- Secure Password Hashing
- SQLite Database Integration

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend Development |
| Flask | Web Framework |
| SQLite | Database |
| DistilBERT | Complaint Classification |
| Sentence Transformers | Semantic Search |
| PyTorch | Deep Learning |
| HTML/CSS | Frontend |
| Flask-Session | Session Management |

---

# System Architecture

```text
User Complaint
       ↓
DistilBERT Classification Model
       ↓
Category Prediction
       ↓
Department Mapping
       ↓
Basic RAG Retrieval
       ↓
Suggested Solution + Complaint Storage
```

---

# Project Modules

## 1. User Module

Users can:

- Register
- Login
- Submit complaints
- Upload images
- Track complaint status
- View AI-generated suggestions

---

## 2. Admin Module

Admins can:

- Login department-wise
- View department complaints
- Track complaint statistics
- Monitor pending complaints

---

## 3. AI Classification Module

The project uses DistilBERT for multi-class complaint classification.

The model predicts categories like:

- Theft
- Power Outage
- Water Leakage
- Potholes
- Internet Down
- Hospital Negligence
- Noise Pollution

and many more.

---

# Model Training

## Dataset

The model was trained using a custom complaint dataset containing thousands of complaint examples with category labels.

---

## Training Workflow

```python
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels
)
```

---

## Prediction Logic

```python
def predict_complaint(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = torch.argmax(outputs.logits).item()

    return label_encoder.inverse_transform([pred_id])[0]
```

---

# Basic RAG Implementation

This project uses a lightweight Retrieval-Augmented Generation approach.

Instead of using a large LLM pipeline, the system retrieves similar complaints using semantic embeddings and returns the most relevant solution.

---

## RAG Workflow

```text
Complaint Query
      ↓
Sentence Embedding
      ↓
Similarity Search
      ↓
Retrieve Similar Complaints
      ↓
Return Best Solution
```

---

## Embedding Model

```python
model = SentenceTransformer('all-MiniLM-L6-v2')
```

---

## Basic RAG Retrieval

```python
def generate_rag_response(query):

    query_emb = model.encode([query], convert_to_tensor=True)

    similarities = (text_embeddings @ query_emb.T).cpu().numpy().flatten()

    top_indices = similarities.argsort()[-3:][::-1]

    similar_cases = []

    for idx in top_indices:
        similar_cases.append({
            "complaint": texts[idx],
            "solution": solutions[idx]
        })

    final_solution = similar_cases[0]["solution"]

    return {
        "final_solution": final_solution,
        "similar_cases": similar_cases
    }
```

---

# Department Routing

The system automatically maps complaint categories to departments.

Example:

```python
CATEGORY_TO_DEPT = {
    "Theft": "police",
    "Power Outage": "electricity",
    "Water Leakage": "water",
    "Potholes": "transport",
    "Internet Down": "telecom"
}
```

---

# Database Design

## Users Table

| Field | Type |
|---|---|
| id | Integer |
| username | Text |
| email | Text |
| password | Text |

---

## Complaints Table

| Field | Type |
|---|---|
| id | Integer |
| user_id | Integer |
| complaint_text | Text |
| category | Text |
| department | Text |
| address | Text |
| latitude | Text |
| longitude | Text |
| image_path | Text |
| status | Text |

---

# Security Features

- Password hashing using Werkzeug
- Session management using Flask-Session
- Secure image uploads
- File type validation

---

# Performance

The system provides:

- Fast complaint classification
- Lightweight RAG retrieval
- Low memory usage
- Real-time department assignment

---

# Future Improvements

- Real-time government API integration
- Complaint priority prediction
- OCR for image complaints
- Voice complaint support
- Multilingual complaint handling
- Advanced LLM-based RAG pipeline
- Geo-based complaint clustering

---


---

# How to Run

Commands

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run model
python model.py

# Start FastAPI server
uvicorn main:app --reload

# Conclusion

This project demonstrates how AI and NLP can be integrated into public complaint systems to automate classification, routing, and response generation.

The combination of DistilBERT and basic RAG makes the system lightweight, practical, and scalable for real-world use cases.

The project focuses on solving real problems with simple and efficient AI implementation rather than using unnecessarily complex architectures.

---
