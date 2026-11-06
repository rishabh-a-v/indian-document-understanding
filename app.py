"""Gradio front-end.   python app.py [--share] [--no-translate]"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import gradio as gr  # noqa: E402

from idu.documents import Document  # noqa: E402
from idu.langs import LANGS, detect_lang  # noqa: E402
from idu.pipeline import DocumentAI, route  # noqa: E402

_STATE = {"ai": None, "docs": {}, "translate": True}
OUT_LANGS = ["Same as document", "English"] + list(LANGS)


def get_ai() -> DocumentAI:
    if _STATE["ai"] is None:
        from idu.models import PatramVLM, Translator, hf_login
        hf_login()
        tr = Translator() if _STATE["translate"] else None
        _STATE["ai"] = DocumentAI(PatramVLM(), tr)
    return _STATE["ai"]


def get_doc(file) -> Document:
    path = file if isinstance(file, str) else getattr(file, "name", None)
    if not path:
        raise gr.Error("Upload a PDF or image first.")
    if path not in _STATE["docs"]:
        _STATE["docs"][path] = Document(path)
    return _STATE["docs"][path]


def out_lang_code(choice, fallback="eng_Latn"):
    if choice == "Same as document":
        return fallback
    return "eng_Latn" if choice == "English" else LANGS[choice]


def analyze(file, page_no, fields_text, out_lang):
    doc, ai = get_doc(file), get_ai()
    page = int(min(max(page_no or 1, 1), doc.n_pages)) - 1
    img = doc.page_image(page)
    fields = [f.strip() for f in (fields_text or "").split(",") if f.strip()] or None
    extracted, _ = ai.extract_fields(img, fields)
    summary = ai.summarize(img)
    tgt = None if out_lang == "Same as document" else out_lang_code(out_lang)
    summary = ai._to_lang(summary, tgt)
    return img, extracted, summary


def ask(file, question, out_lang, k, verify):
    if not (question or "").strip():
        raise gr.Error("Type a question.")
    doc, ai = get_doc(file), get_ai()
    intent = route(question)
    img0 = doc.page_image(0)
    if intent == "summarize":
        return ai.summarize(img0), f"Intent: **summarize** (page 1)", [img0], []
    if intent == "extract":
        fields, _ = ai.extract_fields(img0)
        return json.dumps(fields, ensure_ascii=False, indent=2), "Intent: **extract** (page 1)", [img0], []
    tgt = None if out_lang == "Same as document" else out_lang_code(out_lang)
    res = ai.ask(doc, question, k=int(k), verify=bool(verify), answer_lang=tgt)
    info = (f"Intent: **question answering** | pages searched (best first): "
            f"{[p + 1 for p in res['retrieved']]} | answer from page: "
            f"{(res['page'] + 1) if res['page'] is not None else 'n/a'} | verified: {res['verified']}")
    if doc.warnings:
        info += "\n\n> " + doc.warnings[0]
    pages = [doc.page_image(p) for p in res["retrieved"]]
    return res["answer"] or "NOT FOUND", info, pages, res["trace"]


def build_ui():
    with gr.Blocks(title="Indian Document Understanding") as demo:
        gr.Markdown("# 🇮🇳 Indian Document Understanding Tool\n"
                    "Patram-7B (vision-language) + IndicTrans2 (translation) + page retrieval. "
                    "Upload a notice, form or government document (PDF or image).")
        file = gr.File(label="Document (PDF / image)", file_types=[".pdf", ".png", ".jpg", ".jpeg"])
        out_lang = gr.Dropdown(OUT_LANGS, value="Same as document", label="Output language")
        with gr.Tabs():
            with gr.Tab("Extract & summarize"):
                with gr.Row():
                    page_no = gr.Number(value=1, precision=0, label="Page (1-based)")
                    fields_text = gr.Textbox(label="Fields to extract (comma-separated, optional)",
                                             placeholder="date, reference_number, amount, deadline")
                btn = gr.Button("Analyze", variant="primary")
                preview = gr.Image(label="Page", type="pil")
                fields_out = gr.JSON(label="Extracted fields")
                summary_out = gr.Textbox(label="Summary", lines=6)
                btn.click(analyze, [file, page_no, fields_text, out_lang], [preview, fields_out, summary_out])
            with gr.Tab("Ask the document"):
                q = gr.Textbox(label="Question (any Indian language or English)",
                               placeholder="गांव सनाला में सर्वेक्षण संख्या 100 का क्षेत्रफल कितना है?")
                with gr.Row():
                    k = gr.Slider(1, 5, value=2, step=1, label="Pages to retrieve")
                    verify = gr.Checkbox(value=True, label="Verify answer against the page")
                ask_btn = gr.Button("Answer", variant="primary")
                ans = gr.Textbox(label="Answer", lines=3)
                info = gr.Markdown()
                gallery = gr.Gallery(label="Retrieved pages", columns=3, height=320)
                trace = gr.JSON(label="Agent trace")
                ask_btn.click(ask, [file, q, out_lang, k, verify], [ans, info, gallery, trace])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--no-translate", action="store_true", help="skip loading IndicTrans2")
    a = ap.parse_args()
    _STATE["translate"] = not a.no_translate
    build_ui().queue().launch(share=a.share, debug=False)
