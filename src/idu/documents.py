"""Load PDFs / images, render pages, and get per-page text for retrieval.

Page text comes from OCR (Tesseract, Indic packs) when available. Many Indian
government PDFs use legacy fonts whose embedded text layer decodes to wrong
Devanagari, so the text layer is only a fallback.
"""
import os

from PIL import Image

from .langs import TESSERACT
from .retrieval import PageRetriever

try:
    import pymupdf as _fitz
except ImportError:  # older PyMuPDF
    import fitz as _fitz  # type: ignore

DEFAULT_OCR_LANGS = os.environ.get("IDU_OCR_LANGS", "hin+eng")


def _ocr_langs_available(requested: str):
    """Requested packs that are installed; [] if OCR can't be used."""
    try:
        import pytesseract
        have = set(pytesseract.get_languages(config=""))
    except Exception:
        return []
    langs = [l for l in requested.split("+") if l in have]
    # eng alone cannot read Indic scripts, so treat it as 'no usable OCR'
    return langs if any(l != "eng" for l in langs) else []


class Document:
    def __init__(self, path: str, vlm_max_side: int = 1600, ocr_langs: str = None):
        self.path = path
        self.vlm_max_side = vlm_max_side
        self.ocr_langs = ocr_langs or DEFAULT_OCR_LANGS
        self.warnings = []
        self.is_pdf = path.lower().endswith(".pdf")
        self._pdf = _fitz.open(path) if self.is_pdf else None
        self._img = None if self.is_pdf else Image.open(path).convert("RGB")
        self.n_pages = len(self._pdf) if self.is_pdf else 1
        self._images, self._texts, self._retriever = {}, {}, None
        self.text_source = None

    def _render(self, i: int, dpi: int) -> Image.Image:
        if not self.is_pdf:
            return self._img.copy()
        pix = self._pdf[i].get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    def page_image(self, i: int = 0) -> Image.Image:
        """Page image for the VLM, downscaled to vlm_max_side."""
        if i not in self._images:
            img = self._render(i, 150)
            img.thumbnail((self.vlm_max_side, self.vlm_max_side))
            self._images[i] = img
        return self._images[i]

    def page_text(self, i: int) -> str:
        if i in self._texts:
            return self._texts[i]
        langs = _ocr_langs_available(self.ocr_langs)
        if langs:
            import pytesseract
            text = pytesseract.image_to_string(self._render(i, 200), lang="+".join(langs))
            self.text_source = "ocr"
        else:
            text = self._pdf[i].get_text() if self.is_pdf else ""
            self.text_source = "text_layer"
            msg = (f"OCR unavailable for '{self.ocr_langs}'; using the PDF text layer, which can be "
                   "garbled for legacy Indic fonts. Install tesseract-ocr-hin (apt) for reliable retrieval.")
            if msg not in self.warnings:
                self.warnings.append(msg)
        self._texts[i] = text
        return text

    def retriever(self) -> PageRetriever:
        if self._retriever is None:
            self._retriever = PageRetriever([self.page_text(i) for i in range(self.n_pages)])
        return self._retriever


def ocr_lang_string(codes) -> str:
    """['hin_Deva','eng_Latn'] -> 'hin+eng'."""
    return "+".join(TESSERACT[c] for c in codes if c in TESSERACT)
