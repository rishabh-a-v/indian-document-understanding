"""Run the evaluation: metrics, retrieval hit-rate, failure tagging, report.

    python eval/run_eval.py --cases eval/testcases.jsonl --out eval/results/run1
Outputs: results.jsonl, summary.json, failure_analysis.md
"""
import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from idu.documents import Document  # noqa: E402
from idu.failure import TAGS, tag_failure  # noqa: E402
from idu.metrics import bootstrap_ci, score_text  # noqa: E402


def load_cases(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip() and not line.startswith("#")]


def _golds(case):
    ans = case["answer"]
    golds = [ans] if isinstance(ans, str) else list(ans)
    return golds + list(case.get("aliases", []))


def _row(case, **kw):
    base = {k: case.get(k) for k in ("id", "kind", "doc", "language", "doc_lang", "doc_type", "qtype")}
    base.update(kw)
    return base


def run(cases, ai, out_dir, root=ROOT, k=2, verify=True, use_gold_page=False, doc_factory=Document):
    os.makedirs(out_dir, exist_ok=True)
    docs, rows = {}, []
    for case in cases:
        path = os.path.join(root, case["doc"])
        if path not in docs:
            docs[path] = doc_factory(path)
        doc = docs[path]
        t0 = time.perf_counter()
        if case["kind"] == "qa":
            gold_page = case.get("page")
            res = ai.ask(doc, case["question"], k=k, verify=verify,
                         page=gold_page if use_gold_page else None)
            dt = time.perf_counter() - t0
            golds, pred = _golds(case), res["answer"]
            retrieved = None if use_gold_page else res["retrieved"]
            m = score_text(pred, golds)
            rows.append(_row(case, question=case["question"], gold=golds[0], pred=pred, **m,
                             retrieved=res["retrieved"], gold_page=gold_page, verified=res["verified"],
                             retrieval_hit=(None if (retrieved is None or gold_page is None)
                                            else gold_page in retrieved),
                             latency_s=round(dt, 2),
                             tags=tag_failure(pred, golds, retrieved, gold_page)))
        elif case["kind"] == "extraction":
            gold_fields = {f: v for f, v in case["fields"].items() if v is not None}
            pred_fields, raw = ai.extract_fields(doc.page_image(case.get("page", 0)), list(case["fields"]))
            dt = (time.perf_counter() - t0) / max(len(gold_fields), 1)
            for fname, gold in gold_fields.items():
                golds = [gold] if isinstance(gold, str) else list(gold)
                val = pred_fields.get(fname)
                pred = json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else ("" if val is None else str(val))
                m = score_text(pred, golds)
                rows.append(_row(case, id=f"{case['id']}:{fname}", question=fname, gold=golds[0], pred=pred,
                                 **m, retrieved=None, gold_page=case.get("page", 0), verified=None,
                                 retrieval_hit=None, latency_s=round(dt, 2), tags=tag_failure(pred, golds)))
        else:
            raise ValueError(f"unknown kind {case['kind']!r} in case {case.get('id')}")
    summary = summarize(rows)
    with open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_failure_analysis(rows, summary, os.path.join(out_dir, "failure_analysis.md"))
    return rows, summary


def _agg(rows):
    n = len(rows)
    if not n:
        return {"n": 0}
    em = [r["em"] for r in rows]
    lo, hi = bootstrap_ci(em)
    return {"n": n, "em": round(sum(em) / n, 3), "em_ci95": [round(lo, 3), round(hi, 3)],
            "anls": round(sum(r["anls"] for r in rows) / n, 3),
            "f1": round(sum(r["f1"] for r in rows) / n, 3)}


def summarize(rows):
    s = {"overall": _agg(rows)}
    for kind in sorted({r["kind"] for r in rows}):
        s[f"kind={kind}"] = _agg([r for r in rows if r["kind"] == kind])
    for key in ("language", "doc_type", "qtype", "doc_lang"):
        groups = defaultdict(list)
        for r in rows:
            groups[r.get(key) or "unknown"].append(r)
        s[f"by_{key}"] = {g: _agg(v) for g, v in sorted(groups.items())}
    hits = [r["retrieval_hit"] for r in rows if r.get("retrieval_hit") is not None]
    if hits:
        s["retrieval_hit_rate"] = {"n": len(hits), "hit": round(sum(hits) / len(hits), 3)}
    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    if lat:
        lat_sorted = sorted(lat)
        s["latency_s"] = {"mean": round(statistics.mean(lat), 2), "p50": lat_sorted[len(lat) // 2],
                          "p95": lat_sorted[min(int(0.95 * len(lat)), len(lat) - 1)]}
    s["failure_tags"] = dict(Counter(r["tags"][0] for r in rows if r["tags"]))
    return s


def _table(title, d):
    lines = [f"**{title}**", "", "| group | n | EM | ANLS | token-F1 |", "|---|---|---|---|---|"]
    for g, a in d.items():
        lines.append(f"| {g} | {a['n']} | {a['em']} | {a['anls']} | {a['f1']} |")
    return "\n".join(lines) + "\n"


def write_failure_analysis(rows, summary, path, examples_per_tag=3):
    o = summary["overall"]
    L = ["# Failure analysis", "",
         f"Cases scored: **{o['n']}**  |  EM **{o.get('em')}** (95% CI {o.get('em_ci95')})  |  "
         f"ANLS **{o.get('anls')}**  |  token-F1 **{o.get('f1')}**", ""]
    if "retrieval_hit_rate" in summary:
        L += [f"Page-retrieval hit rate: **{summary['retrieval_hit_rate']['hit']}** "
              f"over {summary['retrieval_hit_rate']['n']} cases with a labelled gold page.", ""]
    for key in ("by_language", "by_doc_type", "by_qtype"):
        L.append(_table(key.replace("by_", "Breakdown by "), summary[key]))
    failures = [r for r in rows if r["tags"]]
    L += ["## Error taxonomy (auto-tagged; review by hand)", "",
          f"{len(failures)} of {len(rows)} predictions are not exact matches.", "",
          "| primary tag | count | meaning |", "|---|---|---|"]
    for tag, cnt in Counter(r["tags"][0] for r in failures).most_common():
        L.append(f"| {tag} | {cnt} | {TAGS[tag]} |")
    L.append("")
    by_tag = defaultdict(list)
    for r in failures:
        by_tag[r["tags"][0]].append(r)
    for tag, rs in sorted(by_tag.items(), key=lambda kv: -len(kv[1])):
        L += [f"### {tag}  ({len(rs)})", ""]
        for r in sorted(rs, key=lambda r: r["anls"])[:examples_per_tag]:
            L += [f"- `{r['id']}` ({r.get('qtype')}, {r.get('language')}) — Q: {r['question']}",
                  f"  - gold: `{r['gold']}`", f"  - pred: `{r['pred']}`"]
            if r.get("retrieved") is not None:
                L.append(f"  - retrieved pages: {r['retrieved']} (gold page: {r.get('gold_page')})")
        L.append("")
    L += ["## Manual analysis (fill in)", "",
          "- Which error types dominate, and why (OCR/vision limits, table structure, language)?",
          "- Does accuracy drop for English questions on Hindi documents, or for table lookups?",
          "- What would you change (higher resolution, better retrieval, prompt, fine-tuning)?", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(ROOT, "eval", "testcases.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "eval", "results", time.strftime("%Y%m%d-%H%M%S")))
    ap.add_argument("--k", type=int, default=2, help="pages retrieved per question")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--gold-page", action="store_true",
                    help="skip retrieval: use the labelled page (isolates VLM accuracy)")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    from idu.models import PatramVLM, hf_login
    from idu.pipeline import DocumentAI
    hf_login()
    cases = load_cases(args.cases)[: args.limit]
    ai = DocumentAI(PatramVLM())
    _, summary = run(cases, ai, args.out, k=args.k, verify=not args.no_verify, use_gold_page=args.gold_page)
    print(json.dumps(summary["overall"], indent=2))
    print("Report:", os.path.join(args.out, "failure_analysis.md"))


if __name__ == "__main__":
    main()
