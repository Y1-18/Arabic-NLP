# Simple Multimodal RAG

A beginner-friendly multimodal Retrieval-Augmented Generation (RAG) project
that lets you search **text and images** from your own files and generate
answers grounded in the retrieved content.

## Features

-  Loads PDFs, `.txt` files, and images from a `data/` folder
-  Chunks text using LangChain's `RecursiveCharacterTextSplitter`
-  Embeds text with **SentenceTransformers** (`jinaai/jina-embeddings-v4`)
-  Embeds images with **CLIP** (`openai/clip-vit-base-patch32`)
-  Stores everything in **FAISS** vector indexes
-  Search by text query → top-k chunks
-  Search by text query → top-k images (cross-modal via CLIP)
-  Generates answers using a HuggingFace LLM (`flan-t5-base` by default)
-  Evaluation with **RAGAS** (faithfulness, answer relevancy, context precision)
  or simple offline metrics if you don't have an API key
- **Gradio UI** with text search, image search, and Q&A tabs

## Project structure

```
multimodal_rag/
├── data/              # put your PDFs / .txt / images here
├── ingest.py          # build the vector store
├── rag.py             # search + answer
├── evaluate.py        # evaluate the RAG
├── app.py             # Gradio UI
├── requirements.txt
└── README.md
```

## Setup

### Local

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. put files into data/
#    (PDFs, .txt, .png, .jpg are supported)

# 4. build the vector store
python ingest.py

# 5. run a demo
python rag.py

# 6. launch the Gradio UI (recommended!)
python app.py
# then open http://127.0.0.1:7860

# 7. (optional) evaluate
python evaluate.py
```

### Google Colab

```python
# 1. clone or upload the project, then:
%cd multimodal_rag
!pip install -r requirements.txt

# 2. upload files into the data/ folder using the file panel
#    (or use !wget / !cp to copy them in)

# 3. build the index
!python ingest.py

# 4. run a demo
!python rag.py

# 5. launch the Gradio UI (use share=True in app.py for a public link)
!python app.py

# 6. evaluate
!python evaluate.py
```

## Example usage (Python)

```python
from rag import MultimodalRAG

rag = MultimodalRAG()

# 1. Text search → returns top-k matching chunks
hits = rag.search_text("What is the main topic of the book?", k=3)
for h in hits:
    print(h["source"], "→", h["text"][:120])

# 2. Image search → text-to-image via CLIP
images = rag.search_image("a page with Arabic poetry", k=3)
for img in images:
    print(img["image_path"], "score:", img["score"])

# 3. Full RAG → retrieve context + LLM-generated answer
result = rag.answer("Who wrote the book?", k=3)
print("Answer:", result["answer"])
print("Sources:", result["sources"])
```

## Evaluation

`evaluate.py` ships with a small test set you can edit at the bottom of the file:

```python
TEST_QUESTIONS = [
    {"query": "What is the main subject of the book?", "ground_truth": "..."},
    {"query": "Who is the author?", "ground_truth": "..."},
]
```

- If `OPENAI_API_KEY` is set in your environment, it uses **RAGAS** for proper
  faithfulness / answer relevancy / context precision metrics.
- Otherwise it falls back to **simple embedding-based metrics** that work fully
  offline (no API key needed).

```bash
# offline metrics (default)
python evaluate.py

# RAGAS metrics
export OPENAI_API_KEY=sk-...
python evaluate.py
```

## Switching the LLM

Edit the top of `rag.py`:

```python
LLM_MODEL = "google/flan-t5-base"   # default — small, free, CPU-friendly
USE_OPENAI = False                  # flip to True to use OpenAI
OPENAI_MODEL = "gpt-4o-mini"
```

For OpenAI, also `pip install openai` (uncomment in `requirements.txt`) and
set `OPENAI_API_KEY`.

## Tips

- `flan-t5-base` is small and runs on CPU, but answers are short. For better
  answers swap it for `google/flan-t5-large` (GPU recommended) or use OpenAI.
- The first run downloads model weights (~500 MB total). Subsequent runs are fast.
- For very large document sets, consider switching `IndexFlatL2` to
  `IndexIVFFlat` in `ingest.py` for faster search.
