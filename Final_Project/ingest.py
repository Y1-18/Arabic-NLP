"""
ingest.py
---------
Build the vector store from data/.

Each text chunk is linked to the page image it came from, so the UI can
show both text and page image together.
"""

import os
import pickle
import re

import faiss
import numpy as np
import torch
from transformers import AutoModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ocr_extract import MistralOCR, rasterize_pdf_pages

DATA_DIR = "data"
INDEX_DIR = "vector_store"
PAGES_DIR = os.path.join(INDEX_DIR, "pages")

EMBED_MODEL = "jinaai/jina-embeddings-v4"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 8


def clean_text(t: str) -> str:
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return clean_text(f.read())


def collect_documents(data_dir: str):
    pdfs, txts = [], []
    for root, _, files in os.walk(data_dir):
        for name in files:
            path = os.path.join(root, name)
            ext = name.lower().split(".")[-1]
            if ext == "pdf":
                pdfs.append(path)
            elif ext == "txt":
                txts.append(path)
    return pdfs, txts


def chunk_pages(pages, source, page_images):
    """Split each page's text into chunks, keeping page_number + image path."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "۔", ".", "،", ",", " ", ""],
    )
    chunks = []
    for entry in pages:
        page_num = entry["page"]
        text = entry["text"]
        if not text:
            continue
        # Get image for this page (1-indexed)
        img_path = page_images[page_num - 1] if page_num - 1 < len(page_images) else None
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if piece:
                chunks.append({
                    "source": source,
                    "page": page_num,
                    "image": img_path,
                    "text": piece,
                })
    return chunks


def load_jina():
    print(f"Loading embedder: {EMBED_MODEL}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        EMBED_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    print(f"Embedder loaded on {device}")
    return model


def _to_numpy(embs):
    if isinstance(embs, list):
        embs = torch.stack([e if torch.is_tensor(e) else torch.tensor(e) for e in embs])
    if torch.is_tensor(embs):
        embs = embs.detach().cpu().numpy()
    return np.asarray(embs, dtype="float32")


def embed_text_chunks(model, chunks):
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} text chunks...")
    embs = model.encode_text(
        texts=texts, task="retrieval", prompt_name="passage", batch_size=BATCH_SIZE,
    )
    return _to_numpy(embs)


def build_faiss_index(vectors):
    if vectors.shape[0] == 0:
        return None
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def save_store(index, metadata, name: str):
    os.makedirs(INDEX_DIR, exist_ok=True)
    if index is not None:
        faiss.write_index(index, os.path.join(INDEX_DIR, f"{name}.faiss"))
    with open(os.path.join(INDEX_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved {name} store ({len(metadata)} items)")


def ocr_cache_path(pdf_path: str) -> str:
    base = os.path.basename(pdf_path)
    return os.path.join(INDEX_DIR, f"{base}.ocr.pkl")


def get_pdf_pages_with_cache(ocr, pdf_path: str) -> list:
    """Return list of {'page': N, 'text': '...'} for a PDF, cached on disk."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    cache = ocr_cache_path(pdf_path)
    if os.path.exists(cache):
        print(f"Using cached OCR for {pdf_path}")
        with open(cache, "rb") as f:
            return pickle.load(f)

    pages = ocr.read_pdf(pdf_path)
    for p in pages:
        p["text"] = clean_text(p["text"])
    with open(cache, "wb") as f:
        pickle.dump(pages, f)
    print(f"Cached OCR → {cache}")
    return pages


def main():
    print(f"Reading from: {DATA_DIR}/")
    pdfs, txts = collect_documents(DATA_DIR)
    print(f"Found {len(pdfs)} PDFs and {len(txts)} text files")

    all_chunks = []

    if pdfs:
        ocr = MistralOCR()
        for pdf in pdfs:
            # 1. OCR (cached)
            pages = get_pdf_pages_with_cache(ocr, pdf)
            # 2. Rasterize pages to JPEG (cached on disk)
            page_images = rasterize_pdf_pages(pdf, PAGES_DIR)
            # 3. Chunk text linked to page images
            source = os.path.basename(pdf)
            chunks = chunk_pages(pages, source, page_images)
            print(f"  {source}: {len(chunks)} chunks")
            all_chunks.extend(chunks)

    for txt in txts:
        text = load_txt(txt)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        )
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if piece:
                all_chunks.append({
                    "source": os.path.basename(txt),
                    "page": None,
                    "image": None,
                    "text": piece,
                })

    if not all_chunks:
        print("No content found. Add PDFs or .txt files to data/")
        return

    sample = all_chunks[0]
    print(f"\n--- Sample chunk ---")
    print(f"Source: {sample['source']}  Page: {sample['page']}")
    print(f"Image:  {sample['image']}")
    print(f"Text:   {sample['text'][:300]}")
    print("---\n")

    jina = load_jina()
    text_vectors = embed_text_chunks(jina, all_chunks)
    print(f"Text embedding shape: {text_vectors.shape}")
    text_index = build_faiss_index(text_vectors)
    save_store(text_index, all_chunks, "text")

    print("\n✅ Ingestion complete. Now run: python app.py")


if __name__ == "__main__":
    main()