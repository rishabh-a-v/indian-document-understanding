"""Task layer: field extraction, summarisation and an agentic QA loop.

QA agent loop (retrieve -> answer -> verify -> retry):
  1. rank pages with the PageRetriever (OCR text, char n-gram TF-IDF)
  2. ask the VLM on the best page image, telling it to say NOT FOUND if absent
  3. ask the VLM to verify its own answer against the same page
  4. if not found / not verified, move to the next page; return the first
     verified answer (else the first non-empty one), with a full trace.
"""
import json
import re

from .langs import detect_lang
from .metrics import is_empty

DEFAULT_FIELDS = ["document_type", "issuing_authority", "date", "reference_number",
                  "name", "address", "amount", "deadline", "subject"]

QA_PROMPT = ("Answer the question using only this document page. Reply with only the answer, as "
             "short as possible, copied exactly from the document. If the answer is not on this "
             "page, reply exactly: NOT FOUND.\nQuestion: {q}")
VERIFY_PROMPT = ("Question: {q}\nProposed answer: {a}\nLook at this document page. Is the proposed "
                 "answer correct and clearly stated on this page? Reply with only Yes or No.")
EXTRACT_PROMPT = ("Read this document carefully. Extract the following fields and return ONLY a valid "
                  "JSON object with exactly these keys: {keys}. Copy values exactly as written in the "
                  "document. If a field is not present, use null.")
SUMMARY_PROMPT = ("Summarize this document in under {n} words. Mention what it is, who issued it, what "
                  "action is required, and any deadlines or amounts.")


def parse_json_loose(raw: str, fields):
    """Pull a JSON object out of model output; fall back to 'key: value' lines."""
    m = re.search(r"\{.*\}", raw or "", flags=re.S)
    if m:
        for cand in (m.group(0), m.group(0).replace("'", '"')):
            try:
                return json.loads(cand)
            except json.JSONDecodeError:
                pass
    out = {}
    for f in fields:
        mm = re.search(rf"{re.escape(f)}\s*[:\-]\s*(.+)", raw or "", flags=re.I)
        out[f] = mm.group(1).strip().strip(",\"'") if mm else None
    return out


def route(text: str) -> str:
    """Tiny intent router: 'summarize' | 'extract' | 'qa' (English + Hindi keywords)."""
    t = text or ""
    if re.search(r"summari[sz]e|summary|सारांश|संक्षेप", t, re.I):
        return "summarize"
    if re.search(r"\bextract\b|\bfields?\b|\bjson\b|फ़?ील्ड|निकाल", t, re.I):
        return "extract"
    return "qa"


class DocumentAI:
    """vlm: callable(image, prompt, max_new_tokens=..) -> str.  translator: optional."""

    def __init__(self, vlm, translator=None):
        self.vlm = vlm
        self.tr = translator

    # ---- single-page tasks -------------------------------------------------
    def extract_fields(self, image, fields=None):
        fields = list(fields or DEFAULT_FIELDS)
        raw = self.vlm(image, EXTRACT_PROMPT.format(keys=", ".join(fields)))
        return parse_json_loose(raw, fields), raw

    def summarize(self, image, max_words: int = 120):
        return self.vlm(image, SUMMARY_PROMPT.format(n=max_words))

    # ---- agentic QA over a whole document -----------------------------------
    def ask(self, doc, question: str, *, k: int = 2, verify: bool = True, page=None,
            answer_lang: str = None):
        if page is not None:
            candidates = [page]
        else:
            candidates = [p for p, _ in doc.retriever().top_k(question, k)]
        trace, best, fallback = [], None, None
        for p in candidates:
            img = doc.page_image(p)
            ans = (self.vlm(img, QA_PROMPT.format(q=question)) or "").strip()
            step = {"page": p, "answer": ans}
            if is_empty(ans):
                step["verdict"] = "not_found"
                trace.append(step)
                continue
            if fallback is None:
                fallback = (p, ans)
            if verify:
                v = (self.vlm(img, VERIFY_PROMPT.format(q=question, a=ans), max_new_tokens=5) or "")
                step["verified"] = v.strip().lower().startswith("yes")
                trace.append(step)
                if step["verified"]:
                    best = (p, ans)
                    break
            else:
                trace.append(step)
                best = (p, ans)
                break
        verified = best is not None and verify
        chosen = best or fallback
        answer = chosen[1] if chosen else ""
        return {
            "answer": self._to_lang(answer, answer_lang),
            "page": chosen[0] if chosen else None,
            "retrieved": candidates,
            "verified": bool(verified),
            "trace": trace,
        }

    def _to_lang(self, text, tgt):
        if not tgt or not self.tr or not text:
            return text
        cur = detect_lang(text)
        if cur == tgt:
            return text
        en = text if cur == "eng_Latn" else self.tr.to_english(text, cur)
        return en if tgt == "eng_Latn" else self.tr.from_english(en, tgt)
