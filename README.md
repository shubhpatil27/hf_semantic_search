# 🔎 Semantic Search with Hugging Face Transformers

A semantic search web application that uses a fine-tuned Sentence Transformer model to retrieve notes based on meaning instead of exact keyword matching.

---

## 🚀 Project Overview

This project demonstrates:

- Transformer-based sentence embeddings
- Fine-tuning a Hugging Face model on custom similarity pairs
- FastAPI backend for serving embeddings and search
- Optional FAISS vector index for fast similarity search
- Simple frontend interface for adding and searching notes

Instead of traditional keyword search, this system performs **semantic similarity search** using cosine similarity between embeddings.

---

## 🧠 How It Works

1. Notes are converted into dense vector embeddings.
2. User query is also converted into an embedding.
3. Cosine similarity is computed between query and note vectors.
4. Top-k most similar notes are returned.

If FAISS is enabled, similarity search is accelerated using a vector index.

---

## 🛠 Tech Stack

- Python 3.11
- Hugging Face `sentence-transformers`
- FastAPI
- FAISS (optional)
- NumPy
- Uvicorn

---

## ⚙️ Setup Instructions (Windows)

```powershell
cd hf_semantic_search
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
cd backend
uvicorn app:app --reload --port 8000

