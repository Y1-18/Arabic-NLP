"""
rag.py
------
Retrieval + generation. Each text result now also carries the page image.
"""

import os
import pickle

import faiss
import numpy as np
import torch
from transformers import AutoModel
from mistralai import Mistral

INDEX_DIR = "vector_store"
EMBED_MODEL = "jinaai/jina-embeddings-v4"
LLM_MODEL = "mistral-small-latest"


class MultimodalRAG:
    def __init__(self):
        print(f"Loading embedder: {EMBED_MODEL}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.embedder = AutoModel.from_pretrained(
            EMBED_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self.embedder.eval()

        print("Loading text store...")
        self.text_index = faiss.read_index(os.path.join(INDEX_DIR, "text.faiss"))
        with open(os.path.join(INDEX_DIR, "text.pkl"), "rb") as f:
            self.text_chunks = pickle.load(f)

        print(f"Loading LLM: {LLM_MODEL}")
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set")
        self.llm_client = Mistral(api_key=api_key)

        print("✅ RAG ready\n")

    def _to_numpy(self, embs):
        if isinstance(embs, list):
            embs = torch.stack([e if torch.is_tensor(e) else torch.tensor(e) for e in embs])
        if torch.is_tensor(embs):
            embs = embs.detach().cpu().numpy()
        return np.asarray(embs, dtype="float32")

    def search(self, query: str, k: int = 3):
        """Returns top-k chunks, each with text + page image."""
        embedding = self.embedder.encode_text(
            texts=[query], task="retrieval", prompt_name="query",
        )
        query_vec = self._to_numpy(embedding)
        distances, indices = self.text_index.search(query_vec, k)

        results = []
        for rank, (i, dist) in enumerate(zip(indices[0], distances[0])):
            if i < 0 or i >= len(self.text_chunks):
                continue
            c = self.text_chunks[i]
            results.append({
                "rank": rank + 1,
                "score": float(dist),
                "source": c["source"],
                "page": c.get("page"),
                "image": c.get("image"),
                "text": c["text"],
            })
        return results

    def _generate(self, prompt: str) -> str:
        response = self.llm_client.chat.complete(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    def answer(self, query: str, k: int = 3):
        retrieved = self.search(query, k=k)
        context = "\n\n".join(
            f"[{r['source']} p.{r['page']}] {r['text']}" for r in retrieved
        )
        prompt = (
            "أجب باستخدام السياق أدناه فقط، وبنفس لغة السؤال. "
            "إذا لم تجد الإجابة في السياق، قل إنك لا تعرف.\n\n"
            f"السياق:\n{context}\n\n"
            f"السؤال: {query}\n"
            "الإجابة:"
        )
        return {
            "query": query,
            "answer": self._generate(prompt),
            "results": retrieved,
        }


def demo():
    rag = MultimodalRAG()
    q = "ما الموضوع الرئيسي للكتاب؟"
    print(f"Query: {q}")
    result = rag.answer(q, k=3)
    print(f"\nAnswer: {result['answer']}\n")
    for r in result["results"]:
        print(f"  [{r['rank']}] {r['source']} p.{r['page']} (score {r['score']:.3f})")
        print(f"      image: {r['image']}")
        print(f"      {r['text'][:120]}...")


if __name__ == "__main__":
    demo()