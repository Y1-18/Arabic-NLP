"""
app.py
------
Gradio UI: one search box → returns text snippets AND page images side by side.
"""

import gradio_client.utils as _gc_utils

_original_get_type = _gc_utils.get_type


def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    try:
        return _original_get_type(schema)
    except Exception:
        return "Any"


_gc_utils.get_type = _safe_get_type

import gradio as gr

from rag import MultimodalRAG

print("Initializing RAG...")
rag = MultimodalRAG()


def do_search(query, k):
    """Return (markdown_text_results, gallery_of_page_images)."""
    if not query.strip():
        return "_الرجاء إدخال سؤال._", []

    hits = rag.search(query, k=int(k))
    if not hits:
        return "_لا توجد نتائج._", []

    # Build markdown for text panel
    lines = [f"### أعلى {len(hits)} نتائج\n"]
    for h in hits:
        snippet = h["text"].replace("\n", " ").strip()
        page_info = f"ص.{h['page']}" if h["page"] else ""
        lines.append(
            f"**[{h['rank']}] `{h['source']}` {page_info}** — score: `{h['score']:.3f}`\n\n"
            f"> {snippet}\n\n---"
        )
    text_md = "\n".join(lines)

    # Build gallery — unique images, keep order
    gallery = []
    seen = set()
    for h in hits:
        img = h["image"]
        if img and img not in seen and os.path.exists(img):
            caption = f"[{h['rank']}] {h['source']} ص.{h['page']}"
            gallery.append((img, caption))
            seen.add(img)

    return text_md, gallery


def do_answer(query, k):
    """Full RAG: returns (answer, sources_md, gallery)."""
    if not query.strip():
        return "_الرجاء إدخال سؤال._", "", []

    result = rag.answer(query, k=int(k))

    src_lines = [f"### السياق المسترجع ({len(result['results'])} مقاطع)\n"]
    gallery = []
    seen = set()
    for r in result["results"]:
        snippet = r["text"].replace("\n", " ").strip()
        page_info = f"ص.{r['page']}" if r["page"] else ""
        src_lines.append(
            f"**[{r['rank']}] `{r['source']}` {page_info}**\n\n> {snippet}\n\n---"
        )
        img = r["image"]
        if img and img not in seen and os.path.exists(img):
            gallery.append((img, f"[{r['rank']}] {r['source']} ص.{r['page']}"))
            seen.add(img)

    return result["answer"], "\n".join(src_lines), gallery


import os

with gr.Blocks(title="Multimodal RAG") as demo:
    gr.Markdown(
        "#  Multimodal RAG\n"
        "اكتب سؤالاً وستظهر لك المقاطع النصية ذات الصلة **مع صور الصفحات** "
        "التي جاءت منها.\n"
    )

    with gr.Tab(" بحث (نص + صور الصفحات)"):
        with gr.Row():
            search_q = gr.Textbox(
                label="السؤال",
                placeholder="ما الأبيات التي قيلت في الصداقة؟",
                scale=4,
            )
            search_k = gr.Slider(1, 10, value=3, step=1, label="عدد النتائج", scale=1)
        search_btn = gr.Button(" بحث", variant="primary")
        with gr.Row():
            with gr.Column(scale=1):
                search_text = gr.Markdown(label="النصوص")
            with gr.Column(scale=1):
                search_gallery = gr.Gallery(
                    label="صور الصفحات", columns=2, height=600
                )
        search_btn.click(do_search, [search_q, search_k], [search_text, search_gallery])
        search_q.submit(do_search, [search_q, search_k], [search_text, search_gallery])

    with gr.Tab(" سؤال + إجابة (RAG)"):
        with gr.Row():
            ask_q = gr.Textbox(
                label="السؤال",
                placeholder="من هو ابن فارس؟",
                scale=4,
            )
            ask_k = gr.Slider(1, 10, value=3, step=1, label="عدد النتائج", scale=1)
        ask_btn = gr.Button("💡 اسأل", variant="primary")
        ask_answer = gr.Textbox(label="الإجابة", lines=4)
        with gr.Row():
            with gr.Column(scale=1):
                ask_sources = gr.Markdown(label="المصادر")
            with gr.Column(scale=1):
                ask_gallery = gr.Gallery(
                    label="صور الصفحات", columns=2, height=600
                )
        ask_btn.click(
            do_answer, [ask_q, ask_k],
            [ask_answer, ask_sources, ask_gallery],
        )
        ask_q.submit(
            do_answer, [ask_q, ask_k],
            [ask_answer, ask_sources, ask_gallery],
        )


if __name__ == "__main__":
    demo.launch(
        share=True,
        show_api=False,
        server_name="0.0.0.0",
        server_port=7860,
    )