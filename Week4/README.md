# 🔍 Arabic RAG System — BBC News

Retrieval-Augmented Generation system for Arabic BBC news articles.

---

## ⚡ Quick Start

```bash
pip install gradio datasets sentence-transformers faiss-cpu numpy transformers
python rag_arabic.py
```

Then open the local URL shown in the terminal.

---

## 🧩 How It Works

```
Your Question
     ↓  MiniLM Embedding
FAISS Search
     ↓  Top-K Chunks
facebook/opt-125m
     ↓
Answer
```

---

## 📐 Two Chunking Methods

**Fixed-Size**
- Splits text into 80-word chunks
- 20-word overlap between chunks
- Consistent, predictable size

**Sentence-Based**
- Groups 3 natural sentences per chunk
- Preserves semantic meaning
- Variable chunk size

---

## 💾 Vector Database

On first run, the app builds two FAISS indexes and saves them to disk:

```
vector_db/
├── fixed.index           ← Fixed-Size FAISS index
├── fixed_chunks.pkl      ← Fixed-Size text chunks
├── sentence.index        ← Sentence-Based FAISS index
└── sentence_chunks.pkl   ← Sentence-Based text chunks
```

On every run after that, indexes load instantly from disk — no rebuilding.

---

## 🤖 Models

| Role | Model | Size |
|------|-------|------|
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` | 120MB |
| Generation | `facebook/opt-125m` | 125MB |

Both run on **CPU**, no GPU required, no API key needed.

---

## 📊 Dataset

[Abdelkareem/arabic-bbc-news](https://huggingface.co/datasets/Abdelkareem/arabic-bbc-news) — 300 articles used.
