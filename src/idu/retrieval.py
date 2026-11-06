"""Page-level retrieval for multi-page documents.

Script-agnostic and model-free: character n-gram TF-IDF (robust to OCR noise and
to Indic scripts) blended with word-level TF-IDF (exact survey numbers, names).
Swap in a dense multilingual embedder (e.g. multilingual-e5) here if you need
cross-lingual retrieval; the interface is just `top_k(query, k)`.
"""
import math

from .metrics import normalize


class PageRetriever:
    """score(page) = TF-IDF blend  +  number_weight * rare-number exact matches
                     + neighbor_weight * TF-IDF blend of the previous page.

    Why the last two terms: in long tables (land records, fee lists, rolls) a section
    header such as "गांव सनाला" appears only on the page where the section starts,
    while the row you ask about may be on a following page. Identifiers (survey /
    reference numbers) are matched exactly and weighted by rarity, and a page
    inherits a share of the previous page's header evidence.
    """

    def __init__(self, page_texts, char_weight: float = 0.6, number_weight: float = 0.5,
                 neighbor_weight: float = 0.5):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.n_pages = len(page_texts)
        self.texts = [normalize(t) for t in page_texts]
        self.char_weight, self.number_weight, self.neighbor_weight = char_weight, number_weight, neighbor_weight
        self._tokens = [set(t.split()) for t in self.texts]
        self.usable = any(self.texts)
        self._char = self._word = None
        if self.usable:
            try:
                self._char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
                self._cm = self._char.fit_transform(self.texts)
                self._word = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True)
                self._wm = self._word.fit_transform(self.texts)
            except ValueError:  # empty vocabulary
                self.usable = False

    def _number_scores(self, q: str):
        nums = [t for t in q.split() if any(c.isdigit() for c in t)]
        if not nums:
            return [0.0] * self.n_pages
        w = {t: math.log((self.n_pages + 1) / (sum(t in s for s in self._tokens) + 1)) + 1.0
             for t in set(nums)}
        total = sum(w.values())
        return [sum(w[t] for t in w if t in toks) / total for toks in self._tokens]

    def scores(self, query: str):
        if not self.usable:
            return [0.0] * self.n_pages
        q = normalize(query)
        cs = (self._cm @ self._char.transform([q]).T).toarray().ravel()
        ws = (self._wm @ self._word.transform([q]).T).toarray().ravel()
        base = self.char_weight * cs + (1 - self.char_weight) * ws
        ns = self._number_scores(q)
        return [float(base[i] + self.number_weight * ns[i]
                      + (self.neighbor_weight * base[i - 1] if i > 0 else 0.0))
                for i in range(self.n_pages)]

    def top_k(self, query: str, k: int = 2):
        """[(page_index, score), ...] best first. Ties / no signal fall back to page order."""
        sc = self.scores(query)
        order = sorted(range(self.n_pages), key=lambda i: (-sc[i], i))
        return [(i, sc[i]) for i in order[:k]]
