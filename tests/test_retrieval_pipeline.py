import os

from idu.documents import Document
from idu.pipeline import DocumentAI, parse_json_loose, route
from idu.retrieval import PageRetriever

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "data", "samples", "gazette_ka_aa_5830_2026.pdf")

PAGES = [
    "सड़क परिवहन और राजमार्ग मंत्रालय अधिसूचना गुजरात जामनगर राष्ट्रीय राजमार्ग टोल प्लाजा",
    "गांव सनाला सर्वेक्षण संख्या 100 0.2095 बुंसा घनश्याम परसोतम सर्वेक्षण संख्या 12 नितिनभाई",
    "गांव छतर सर्वेक्षण संख्या 1024 मंजुलाबेन भीमा सर्वेक्षण संख्या 17 1.3459 सरकारी",
    "गांव अरला सर्वेक्षण संख्या 104 0.0019 सर्वेक्षण संख्या 57 0.7871 सरकारी खराबो",
]


def test_retriever_finds_the_right_page():
    r = PageRetriever(PAGES)
    assert r.top_k("गांव छतर में सर्वेक्षण संख्या 1024 का भूस्वामी कौन है?", 1)[0][0] == 2
    assert r.top_k("गांव अरला सर्वेक्षण संख्या 57 क्षेत्रफल", 1)[0][0] == 3
    assert r.top_k("टोल प्लाजा किस राजमार्ग पर है", 1)[0][0] == 0


def test_table_continuation_page_without_header_is_found():
    # village header only on page 1; the asked row (survey 414) is on page 2, which has no header
    pages = [
        "अधिसूचना भूमि अर्जन राजमार्ग जामनगर",
        "तालुक का नाम कालावड गांव का नाम सनाला 1 10 निजी 0.0239 मोहनभाई 2 100 निजी 0.2095 बुंसा",
        "9 411 निजी 0.2154 सुरजबा 10 413 निजी 0.2713 हंसाबा 11 414 निजी 0.0945 अनुभा पुनाभा",
        "गांव का नाम अरला 94 10 निजी 0.014 केशवजी 95 104 सरकारी 0.0019",
    ] + ["अन्य पृष्ठ"] * 3
    r = PageRetriever(pages)
    top2 = [p for p, _ in r.top_k("गांव सनाला में सर्वेक्षण संख्या 414 का भूस्वामी कौन है?", 2)]
    assert 2 in top2 and 1 in top2


def test_retriever_no_text_falls_back_to_page_order():
    r = PageRetriever(["", "", ""])
    assert [p for p, _ in r.top_k("anything", 2)] == [0, 1]


class FakeImg:
    def __init__(self, pid):
        self.pid = pid


class FakeDoc:
    def __init__(self, pages=PAGES):
        self.n_pages, self._r, self.warnings = len(pages), PageRetriever(pages), []

    def retriever(self):
        return self._r

    def page_image(self, i):
        return FakeImg(i)


def make_vlm(answer_page, answer="0.0019", verify_ok_pages=None):
    """Fake VLM: knows the answer only on `answer_page`; 'verifies' pages in verify_ok_pages."""
    verify_ok_pages = {answer_page} if verify_ok_pages is None else verify_ok_pages

    def vlm(img, prompt, max_new_tokens=512):
        if "Yes or No" in prompt:
            return "Yes" if img.pid in verify_ok_pages else "No"
        return answer if img.pid == answer_page else "NOT FOUND"
    return vlm


Q = "गांव अरला सर्वेक्षण संख्या 104 क्षेत्रफल"


def test_ask_first_retrieved_page_correct():
    res = DocumentAI(make_vlm(3)).ask(FakeDoc(), Q, k=2)
    assert res["answer"] == "0.0019" and res["page"] == 3 and res["verified"] and res["retrieved"][0] == 3


def test_ask_retries_next_page_when_not_found():
    # answer lives on page 3, but force retrieval to try page 1 first via k and a misleading query
    res = DocumentAI(make_vlm(1, "0.2095")).ask(FakeDoc(), Q, k=4)
    assert res["answer"] == "0.2095" and res["page"] == 1
    assert [s["page"] for s in res["trace"]][-1] == 1
    assert res["trace"][0].get("verdict") == "not_found"


def test_ask_unverified_falls_back_to_first_answer():
    res = DocumentAI(make_vlm(3, verify_ok_pages=set())).ask(FakeDoc(), Q, k=2)
    assert res["answer"] == "0.0019" and res["verified"] is False


def test_ask_all_not_found():
    res = DocumentAI(make_vlm(99)).ask(FakeDoc(), Q, k=3)
    assert res["answer"] == "" and res["page"] is None and res["verified"] is False


def test_ask_gold_page_skips_retrieval():
    res = DocumentAI(make_vlm(2, "x")).ask(FakeDoc(), Q, page=2)
    assert res["retrieved"] == [2] and res["answer"] == "x"


def test_route_and_json_parse():
    assert route("Summarize this document") == "summarize"
    assert route("इस दस्तावेज़ का सारांश दें") == "summarize"
    assert route("extract all fields as json") == "extract"
    assert route("सर्वेक्षण संख्या 100 का क्षेत्रफल?") == "qa"
    assert parse_json_loose('text {"a": "1", "b": null} tail', ["a", "b"]) == {"a": "1", "b": None}
    assert parse_json_loose("a: x\nb: y", ["a", "b"]) == {"a": "x", "b": "y"}


def test_real_pdf_loads_renders_and_has_text_fallback():
    doc = Document(SAMPLE)
    assert doc.n_pages == 13
    img = doc.page_image(0)
    assert max(img.size) <= 1600 and min(img.size) > 500
    assert isinstance(doc.page_text(0), str)
    assert doc.text_source in ("ocr", "text_layer")
