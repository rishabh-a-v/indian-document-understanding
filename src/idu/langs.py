"""Language / script helpers shared by the pipeline, retrieval and evaluation."""
import re

# IndicTrans2 (FLORES-style) codes
LANGS = {
    "Hindi": "hin_Deva", "Marathi": "mar_Deva", "Bengali": "ben_Beng",
    "Tamil": "tam_Taml", "Telugu": "tel_Telu", "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym", "Gujarati": "guj_Gujr", "Punjabi": "pan_Guru",
    "Odia": "ory_Orya", "Assamese": "asm_Beng", "Urdu": "urd_Arab",
}

# IndicTrans2 code -> Tesseract language pack
TESSERACT = {
    "hin_Deva": "hin", "mar_Deva": "mar", "ben_Beng": "ben", "tam_Taml": "tam",
    "tel_Telu": "tel", "kan_Knda": "kan", "mal_Mlym": "mal", "guj_Gujr": "guj",
    "pan_Guru": "pan", "ory_Orya": "ori", "asm_Beng": "asm", "urd_Arab": "urd",
    "eng_Latn": "eng",
}

# Unicode block -> code. Script-level guess only: all Devanagari maps to Hindi
# (Marathi/Sanskrit/Nepali share the script), Bengali script maps to Bengali.
SCRIPT_RANGES = [
    (0x0900, 0x097F, "hin_Deva"), (0x0980, 0x09FF, "ben_Beng"),
    (0x0A00, 0x0A7F, "pan_Guru"), (0x0A80, 0x0AFF, "guj_Gujr"),
    (0x0B00, 0x0B7F, "ory_Orya"), (0x0B80, 0x0BFF, "tam_Taml"),
    (0x0C00, 0x0C7F, "tel_Telu"), (0x0C80, 0x0CFF, "kan_Knda"),
    (0x0D00, 0x0D7F, "mal_Mlym"), (0x0600, 0x06FF, "urd_Arab"),
]


def detect_lang(text: str) -> str:
    """Dominant Indian script as an IndicTrans2 code, or 'eng_Latn' if none found."""
    counts = {}
    for ch in text or "":
        cp = ord(ch)
        for lo, hi, code in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts:
        return "eng_Latn"
    return max(counts, key=counts.get)


def split_sentences(text: str):
    parts = re.split(r"(?<=[.!?।॥])\s+|\n+", text or "")
    return [p.strip() for p in parts if p and p.strip()]
