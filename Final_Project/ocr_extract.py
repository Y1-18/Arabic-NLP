"""
ocr_extract.py
---------------
Mistral OCR + per-page image extraction.

Each PDF page is also saved as a JPEG in vector_store/pages/ so the UI can
display the page image next to text results.
"""

import os
import base64
import io
import time

from mistralai import Mistral

OCR_MODEL = "mistral-ocr-latest"
MAX_RETRIES = 3
RETRY_DELAY = 5


class MistralOCR:
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set.")
        print(f"Initializing Mistral OCR client (model: {OCR_MODEL})")
        self.client = Mistral(api_key=api_key)

    def read_pdf(self, pdf_path: str) -> list:
        """OCR a PDF and return a list of dicts:
            [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
        """
        print(f"Reading PDF: {pdf_path}")

        with open(pdf_path, "rb") as f:
            pdf_b64 = base64.b64encode(f.read()).decode("utf-8")
        document_url = f"data:application/pdf;base64,{pdf_b64}"

        response = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"  Sending to Mistral OCR (attempt {attempt}/{MAX_RETRIES})...")
                response = self.client.ocr.process(
                    model=OCR_MODEL,
                    document={"type": "document_url", "document_url": document_url},
                )
                break
            except Exception as e:
                print(f"  Attempt {attempt} failed: {type(e).__name__}: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    raise

        pages = response.pages if hasattr(response, "pages") else []
        print(f"  Got {len(pages)} pages back")

        results = []
        for i, page in enumerate(pages):
            text = getattr(page, "markdown", None) or getattr(page, "text", "") or ""
            results.append({"page": i + 1, "text": text.strip()})
            print(f"  page {i + 1}: {len(text)} chars")
        return results


def rasterize_pdf_pages(pdf_path: str, output_dir: str, dpi: int = 120) -> list:
    """Convert each PDF page to a JPEG image. Returns list of image paths."""
    import fitz  # PyMuPDF

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    doc = fitz.open(pdf_path)
    paths = []
    print(f"Rasterizing {len(doc)} pages of {pdf_path}...")
    for i, page in enumerate(doc):
        out_path = os.path.join(output_dir, f"{base}_p{i + 1:03d}.jpg")
        if not os.path.exists(out_path):
            pix = page.get_pixmap(matrix=mat)
            pix.save(out_path)
        paths.append(out_path)
    doc.close()
    return paths