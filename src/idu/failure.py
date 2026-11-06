"""Heuristic failure taxonomy used to auto-tag wrong predictions.

Tags are suggestions for the error analysis: review them by hand before you
put numbers in a report.
"""
from .langs import detect_lang
from .metrics import (_as_list, best_similarity, exact, is_empty, normalize,
                      numbers, precision_recall)

TAGS = {
    "RETRIEVAL_MISS": "The page that holds the answer was not among the pages retrieved.",
    "EMPTY_OR_NOT_FOUND": "Model returned nothing / null / NOT FOUND.",
    "WRONG_LANGUAGE_OR_SCRIPT": "Answer is in a different script/language than the gold answer.",
    "NUMERIC_MISSING": "Gold contains numbers but the prediction contains none.",
    "NUMERIC_WRONG": "Prediction has numbers, but not the gold ones (digit error / wrong row).",
    "NEAR_MISS_CHAR_ERROR": "Almost right (similarity >= 0.8): matra / spelling / OCR-style character errors.",
    "PARTIAL_ANSWER": "Part of the gold answer is missing (e.g. only some owners / words).",
    "EXTRA_TEXT": "Gold answer is contained but the prediction adds extra words.",
    "WRONG_ANSWER": "Non-empty answer with little overlap: wrong row/entity or hallucination.",
}


def tag_failure(pred, golds, retrieved_pages=None, gold_page=None):
    """Return the list of matching tags (first = primary). Empty list if exact match."""
    golds = _as_list(golds)
    if exact(pred, golds):
        return []
    tags = []
    if retrieved_pages is not None and gold_page is not None and gold_page not in retrieved_pages:
        tags.append("RETRIEVAL_MISS")
    if is_empty(pred):
        tags.append("EMPTY_OR_NOT_FOUND")
        return tags

    # language check only applies when the gold answer actually contains letters
    # (a purely numeric gold like "0.2095" has no language)
    lettered = [g for g in golds if any(ch.isalpha() for ch in g)]
    if lettered and detect_lang(str(pred)) not in {detect_lang(g) for g in lettered}:
        tags.append("WRONG_LANGUAGE_OR_SCRIPT")

    pn = set(numbers(pred))
    gn = [set(numbers(g)) for g in golds if numbers(g)]
    if gn and not any(g <= pn for g in gn):
        tags.append("NUMERIC_MISSING" if not pn else "NUMERIC_WRONG")

    if best_similarity(pred, golds) >= 0.8:
        tags.append("NEAR_MISS_CHAR_ERROR")

    best = max((precision_recall(pred, g) for g in golds), key=lambda pr: pr[0] + pr[1])
    p, r = best
    if r >= 0.25 and p >= 0.8:
        tags.append("PARTIAL_ANSWER")
    elif r >= 0.8 and p < 0.8:
        tags.append("EXTRA_TEXT")
    if not any(t in tags for t in ("NUMERIC_MISSING", "NUMERIC_WRONG", "NEAR_MISS_CHAR_ERROR",
                                   "PARTIAL_ANSWER", "EXTRA_TEXT")):
        tags.append("WRONG_ANSWER")
    return tags
