"""Text normalisation and scoring (EM, ANLS, token-F1) for Indic text."""
import random
import re
import unicodedata

# Indic / Arabic-Indic digit blocks -> ASCII digits
_DIGIT_BASES = [0x0660, 0x06F0, 0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6,
                0x0C66, 0x0CE6, 0x0D66]
_DIGIT_MAP = {base + i: ord("0") + i for base in _DIGIT_BASES for i in range(10)}
_ZERO_WIDTH = dict.fromkeys([0x200B, 0x200C, 0x200D, 0xFEFF], None)
_PUNCT = re.compile(r"[\.,;:!?\(\)\[\]\{\}\"'“”‘’।॥|/\\\-–—_]+")
_EMPTY = {"", "null", "none", "n/a", "na", "nil", "-", "not found", "unknown"}


def normalize(s) -> str:
    """NFC, lower-case, ASCII digits, no zero-width chars / punctuation.
    Decimal points inside numbers (0.2095) are kept."""
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFC", s).translate(_DIGIT_MAP).translate(_ZERO_WIDTH).lower()
    s = re.sub(r"(?<=\d)\.(?=\d)", "\x00", s)
    s = _PUNCT.sub(" ", s)
    s = s.replace("\x00", ".")
    return re.sub(r"\s+", " ", s).strip()


def is_empty(pred) -> bool:
    n = normalize(pred)
    return n in _EMPTY or n.startswith("not found")


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """1 - normalised Levenshtein distance on normalised strings."""
    a, b = normalize(a), normalize(b)
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


def _as_list(golds):
    return [golds] if isinstance(golds, str) else list(golds)


def best_similarity(pred, golds) -> float:
    return max(similarity(pred, g) for g in _as_list(golds))


def anls(pred, golds, threshold: float = 0.5) -> float:
    """DocVQA ANLS for one sample: similarity if NL < threshold else 0."""
    sim = best_similarity(pred, golds)
    return sim if (1.0 - sim) < threshold else 0.0


def exact(pred, golds) -> float:
    p = normalize(pred)
    return float(any(p == normalize(g) for g in _as_list(golds)))


def _f1(pred_toks, gold_toks) -> float:
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = 0
    pool = list(gold_toks)
    for t in pred_toks:
        if t in pool:
            pool.remove(t)
            common += 1
    if common == 0:
        return 0.0
    p, r = common / len(pred_toks), common / len(gold_toks)
    return 2 * p * r / (p + r)


def token_f1(pred, golds) -> float:
    pt = normalize(pred).split()
    return max(_f1(pt, normalize(g).split()) for g in _as_list(golds))


def precision_recall(pred, gold):
    """Bag-of-token precision/recall of pred vs a single gold string."""
    pt, gt = normalize(pred).split(), normalize(gold).split()
    if not pt or not gt:
        return 0.0, 0.0
    pool, common = list(gt), 0
    for t in pt:
        if t in pool:
            pool.remove(t)
            common += 1
    return common / len(pt), common / len(gt)


def numbers(s):
    return re.findall(r"\d+(?:\.\d+)?", normalize(s))


def score_text(pred, golds) -> dict:
    return {"em": exact(pred, golds), "anls": anls(pred, golds), "f1": token_f1(pred, golds)}


def bootstrap_ci(values, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of the mean. Returns (low, high)."""
    values = list(values)
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    m = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(m)) / m for _ in range(n))
    return means[int(alpha / 2 * n)], means[max(int((1 - alpha / 2) * n) - 1, 0)]
