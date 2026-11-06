import json
import os

import run_eval
from csv_to_jsonl import convert
from idu.pipeline import DocumentAI
from test_retrieval_pipeline import PAGES, FakeDoc, make_vlm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_starter_testcases_are_well_formed():
    import pymupdf
    cases = run_eval.load_cases(os.path.join(ROOT, "eval", "testcases.jsonl"))
    assert 30 <= len(cases) <= 50
    assert len({c["doc"] for c in cases}) >= 4
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for c in cases:
        assert os.path.exists(os.path.join(ROOT, c["doc"]))
        assert 0 <= c["page"] < pymupdf.open(os.path.join(ROOT, c["doc"])).page_count
        assert c["kind"] in ("qa", "extraction")
        assert ("question" in c and "answer" in c) if c["kind"] == "qa" else bool(c["fields"])
        assert isinstance(c.get("page"), int)


class FakeAI(DocumentAI):
    def extract_fields(self, image, fields=None):
        return {"state": "गुजरात", "district": "Jamnagar", "taluk": None}, "raw"


def test_run_writes_report_and_scores(tmp_path):
    cases = [
        {"id": "a", "kind": "qa", "doc": "d", "page": 3, "question": "गांव अरला सर्वेक्षण संख्या 104 क्षेत्रफल",
         "answer": "0.0019", "language": "hi", "doc_type": "gaz", "qtype": "table_lookup"},
        {"id": "b", "kind": "qa", "doc": "d", "page": 1, "question": "गांव अरला सर्वेक्षण संख्या 57 क्षेत्रफल",
         "answer": "0.7871", "language": "hi", "doc_type": "gaz", "qtype": "table_lookup"},   # wrong page -> miss
        {"id": "e", "kind": "extraction", "doc": "d", "page": 1, "fields": {"state": "गुजरात", "district": "जामनगर", "taluk": None},
         "language": "hi", "doc_type": "gaz", "qtype": "field_extraction"},
    ]
    ai = FakeAI(make_vlm(3, "0.0019"))
    rows, summary = run_eval.run(cases, ai, str(tmp_path), k=1, doc_factory=lambda p: FakeDoc(PAGES))
    by_id = {r["id"]: r for r in rows}
    assert by_id["a"]["em"] == 1.0 and by_id["a"]["retrieval_hit"] is True
    assert by_id["b"]["em"] == 0.0 and by_id["b"]["tags"][0] == "RETRIEVAL_MISS"
    assert by_id["e:state"]["em"] == 1.0
    assert by_id["e:district"]["tags"][0] == "WRONG_LANGUAGE_OR_SCRIPT"
    assert "e:taluk" not in by_id                      # null gold fields are not scored
    assert summary["overall"]["n"] == 4
    assert summary["retrieval_hit_rate"] == {"n": 2, "hit": 0.5}
    for name in ("results.jsonl", "summary.json", "failure_analysis.md"):
        assert (tmp_path / name).exists()
    report = (tmp_path / "failure_analysis.md").read_text(encoding="utf-8")
    assert "RETRIEVAL_MISS" in report and "Manual analysis" in report


def test_csv_to_jsonl(tmp_path):
    src, dst = tmp_path / "c.csv", tmp_path / "c.jsonl"
    src.write_text(
        "id,kind,doc,page,question,answer,aliases,fields_json,language,doc_lang,doc_type,qtype\n"
        'q1,qa,docs/a.pdf,0,"तारीख क्या है?",27 अक्टूबर 2026,27/10/2026 || 27-10-2026,,hi,hi,notice,date\n'
        'e1,extraction,docs/a.pdf,0,,,,"{""state"": ""गुजरात""}",hi,hi,notice,field_extraction\n',
        encoding="utf-8")
    assert convert(str(src), str(dst)) == 2
    lines = [json.loads(l) for l in dst.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["aliases"] == ["27/10/2026", "27-10-2026"] and lines[0]["page"] == 0
    assert lines[1]["fields"] == {"state": "गुजरात"}
