"""
نظام RAG للأخبار العربية من BBC

"""

import os
import re
import json
import pickle
import gradio as gr
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ============================================================
# المسارات — كل ملفات الـ vector DB تُحفظ هنا
# ============================================================
DB_DIR        = "vector_db"
FIXED_INDEX   = os.path.join(DB_DIR, "fixed.index")
FIXED_CHUNKS  = os.path.join(DB_DIR, "fixed_chunks.pkl")
SENT_INDEX    = os.path.join(DB_DIR, "sentence.index")
SENT_CHUNKS   = os.path.join(DB_DIR, "sentence_chunks.pkl")

os.makedirs(DB_DIR, exist_ok=True)

# ============================================================
# 1. نموذج Embedding خفيف (يعمل على CPU)
# ============================================================
print(" تحميل نموذج Embedding...")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print(" نموذج Embedding جاهز")

# ============================================================
# 2. نموذج التوليد — google/flan-t5-base (250MB، يعمل على CPU)
#    يفهم العربية بشكل معقول ويولّد إجابات مختصرة
# ============================================================
print(" تحميل نموذج التوليد (facebook/opt-125m)...")
generator = pipeline(
    "text-generation",
    model="facebook/opt-125m",
    device=-1,   # -1 = CPU
)
print(" نموذج التوليد جاهز")

# ============================================================
# 3. طريقتا التقسيم Chunking
# ============================================================

def fixed_size_chunking(text, chunk_size=80, overlap=20):
    """تقسيم بحجم ثابت من الكلمات مع تداخل للحفاظ على السياق"""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks

def sentence_chunking(text, sentences_per_chunk=3):
    """تجميع جمل طبيعية — يحافظ على الوحدات الدلالية"""
    sentences = re.split(r'[.!?،؛\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i: i + sentences_per_chunk])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# ============================================================
# 4. بناء أو تحميل الفهارس
# ============================================================

def build_and_save_index(chunks, index_path, chunks_path):
    """بناء FAISS index وحفظه على القرص"""
    print(f"  🔢 تحويل {len(chunks):,} chunk إلى embeddings...")
    embeddings = embed_model.encode(chunks, batch_size=32, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    # الحفظ
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"   تم الحفظ: {index_path}")
    return index, chunks

def load_index(index_path, chunks_path):
    """تحميل FAISS index من القرص"""
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"   تم التحميل: {index_path} ({len(chunks):,} chunk)")
    return index, chunks

# تحقق هل الفهارس موجودة مسبقاً
fixed_exists = os.path.exists(FIXED_INDEX) and os.path.exists(FIXED_CHUNKS)
sent_exists  = os.path.exists(SENT_INDEX)  and os.path.exists(SENT_CHUNKS)

if fixed_exists and sent_exists:
    #  تحميل من القرص — لا حاجة لإعادة البناء
    print("\n تحميل الفهارس من القرص...")
    fixed_index, fixed_chunks   = load_index(FIXED_INDEX, FIXED_CHUNKS)
    sentence_index, sentence_chunks = load_index(SENT_INDEX,  SENT_CHUNKS)
    print(" تم تحميل الفهارس بنجاح")
else:
    #  البناء لأول مرة وحفظ على القرص
    print("\n تحميل بيانات BBC العربية...")
    dataset = load_dataset("Abdelkareem/arabic-bbc-news", split="train")
    print(f"   الأعمدة: {dataset.column_names}")

    text_col = next(
        (c for c in ["content", "text", "article", "body", "description"] if c in dataset.column_names),
        dataset.column_names[0]
    )
    print(f"   العمود المستخدم: '{text_col}'")
    articles = [str(row[text_col]) for row in dataset if row.get(text_col) and str(row[text_col]).strip()][:300]
    print(f" {len(articles)} خبر محمّل")

    print("\n  بناء فهرس Fixed-Size...")
    fixed_chunks = []
    for art in articles:
        fixed_chunks.extend(fixed_size_chunking(art))
    fixed_index, fixed_chunks = build_and_save_index(fixed_chunks, FIXED_INDEX, FIXED_CHUNKS)

    print("\n  بناء فهرس Sentence-Based...")
    sentence_chunks = []
    for art in articles:
        sentence_chunks.extend(sentence_chunking(art))
    sentence_index, sentence_chunks = build_and_save_index(sentence_chunks, SENT_INDEX, SENT_CHUNKS)

print(f"\nFixed: {len(fixed_chunks):,} | Sentence: {len(sentence_chunks):,}")
print(f" ملفات DB محفوظة في: ./{DB_DIR}/")

# ============================================================
# 5. الاسترجاع
# ============================================================

def retrieve(query, method, top_k):
    vec = embed_model.encode([query], batch_size=1)
    vec = np.array(vec, dtype="float32")
    faiss.normalize_L2(vec)
    if method == "fixed":
        scores, indices = fixed_index.search(vec, top_k)
        chunks = fixed_chunks
    else:
        scores, indices = sentence_index.search(vec, top_k)
        chunks = sentence_chunks
    return [(float(s), chunks[i]) for s, i in zip(scores[0], indices[0]) if i >= 0]

# ============================================================
# 6. التوليد بـ flan-t5-base
# ============================================================

def generate_answer(query, results):
    # نبني prompt مختصر — opt-125m صغير جداً
    context = " ".join([chunk[:150] for _, chunk in results[:3]])
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"

    out = generator(
        prompt,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    # نستخرج فقط الجزء بعد "Answer:"
    full_text = out[0]["generated_text"]
    answer = full_text.split("Answer:")[-1].strip()

    # نعرض السياق العربي المسترجع بجانب الإجابة
    arabic_context = "\n".join([f"• {chunk[:300]}" for _, chunk in results[:3]])
    return f"** الإجابة (opt-125m):**\n{answer}\n\n---\n**📰 المقاطع المسترجعة:**\n{arabic_context}"

# ============================================================
# 7. الدالة الرئيسية
# ============================================================

def rag_pipeline(query, method_label, top_k):
    if not query.strip():
        return " الرجاء إدخال سؤال", "", ""

    method  = "fixed" if "Fixed" in method_label else "sentence"
    results = retrieve(query, method, int(top_k))

    # عرض الـ chunks
    chunks_md = ""
    for i, (score, chunk) in enumerate(results, 1):
        filled = int(score * 20)
        bar = "█" * filled + "░" * (20 - filled)
        chunks_md += f"**[{i}] درجة التشابه: `{score:.4f}`**\n`{bar}`\n\n> {chunk}\n\n---\n"

    # الإجابة
    try:
        answer = generate_answer(query, results)
    except Exception as e:
        answer = f" خطأ في التوليد: {e}"

    # الإحصائيات
    total      = len(fixed_chunks) if method == "fixed" else len(sentence_chunks)
    score_vals = [s for s, _ in results]
    db_files   = "\n".join([f"- `{f}`" for f in os.listdir(DB_DIR)])
    stats_md   = f"""
###  إحصائيات البحث

| المعيار | القيمة |
|---------|--------|
| طريقة التقسيم | **{method_label}** |
| إجمالي Chunks في الفهرس | **{total:,}** |
| النتائج المسترجعة | **{len(results)}** |
| أعلى تشابه | **{max(score_vals):.4f}** |
| أدنى تشابه | **{min(score_vals):.4f}** |
| المتوسط | **{np.mean(score_vals):.4f}** |

---

### 💾 ملفات Vector DB المحفوظة (`{DB_DIR}/`)
{db_files}

---

### 🧩 مقارنة الطريقتين

| | Fixed-Size | Sentence-Based |
|-|-----------|----------------|
| وحدة التقسيم | 80 كلمة | 3 جمل |
| تداخل | ✅ 20 كلمة | ❌ لا يوجد |
| حجم ثابت | ✅ | ❌ يتغير |
| يحافظ على المعنى | جزئياً | ✅ أفضل |
"""
    return answer, chunks_md, stats_md

# ============================================================
# 8. واجهة Gradio (متوافقة مع Gradio 6)
# ============================================================

with gr.Blocks(title="RAG - أخبار BBC العربية") as demo:

    gr.Markdown("""
    #  نظام RAG للأخبار العربية — BBC News
    >  **Embedding:** `paraphrase-multilingual-MiniLM-L12-v2`  
    >  **التوليد:** `facebook/opt-125m` (125MB، CPU، بدون API key)  
    >  **Vector DB:** محفوظة في `./vector_db/` — تُحمَّل فوراً في التشغيلات التالية  
    >  **البحث:** FAISS IndexFlatIP (Cosine Similarity)
    """)

    with gr.Row():
        with gr.Column(scale=2):
            query_box = gr.Textbox(
                label=" اكتب سؤالك",
                placeholder="مثال: ما هي أحدث أخبار الاقتصاد؟",
                lines=2,
                rtl=True
            )
            with gr.Row():
                method_radio = gr.Radio(
                    choices=["Fixed-Size Chunking", "Sentence-Based Chunking"],
                    value="Fixed-Size Chunking",
                    label=" طريقة التقسيم"
                )
                top_k_slider = gr.Slider(1, 10, value=5, step=1, label=" عدد النتائج")
            search_btn = gr.Button(" بحث وأجب", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("""
            ### ⚙️ كيف يعمل؟
            ```
            سؤالك
               ↓ MiniLM Embedding
            FAISS Search
               ↓ Top-K Chunks
            flan-t5-base
               ↓
            الإجابة
            ```
            ### 💾 حفظ DB
            أول تشغيل → يبني ويحفظ  
            التشغيلات التالية → يحمّل مباشرة ⚡
            """)

    with gr.Tabs():
        with gr.Tab(" الإجابة"):
            answer_out = gr.Markdown()
        with gr.Tab(" Chunks المسترجعة"):
            chunks_out = gr.Markdown()
        with gr.Tab(" إحصائيات و DB"):
            stats_out = gr.Markdown()

    gr.Examples(
        examples=[
            ["ما هي أحدث الأخبار الاقتصادية؟", "Fixed-Size Chunking", 5],
            ["ما التطورات في مجال التكنولوجيا؟", "Sentence-Based Chunking", 4],
            ["ما آخر أخبار الرياضة؟", "Fixed-Size Chunking", 3],
            ["ما الأخبار المتعلقة بالبيئة والمناخ؟", "Sentence-Based Chunking", 5],
        ],
        inputs=[query_box, method_radio, top_k_slider],
        label=" أمثلة"
    )

    search_btn.click(
        fn=rag_pipeline,
        inputs=[query_box, method_radio, top_k_slider],
        outputs=[answer_out, chunks_out, stats_out]
    )

    gr.Markdown("---\n*درجة التشابه: 1.0 = تطابق تام | 0.0 = لا علاقة*")

if __name__ == "__main__":
    demo.launch(share=True)