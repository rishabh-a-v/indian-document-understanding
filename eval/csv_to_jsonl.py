"""Label test cases in a spreadsheet, then convert to JSONL.

CSV columns (header row required):
  id, kind, doc, page, question, answer, aliases, fields_json, language, doc_lang, doc_type, qtype
- kind: qa | extraction.  page: 0-based page that holds the answer.
- aliases: extra accepted answers separated by ' || '.
- fields_json (extraction only): {"date": "27 अक्टूबर, 2026", "state": "गुजरात"}

    python eval/csv_to_jsonl.py my_cases.csv eval/testcases.jsonl
"""
import csv
import json
import sys


def convert(src, dst):
    out = []
    with open(src, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            case = {"id": row["id"].strip(), "kind": row["kind"].strip(), "doc": row["doc"].strip()}
            if row.get("page", "").strip() != "":
                case["page"] = int(row["page"])
            if case["kind"] == "qa":
                case["question"] = row["question"].strip()
                case["answer"] = row["answer"].strip()
            else:
                case["fields"] = json.loads(row["fields_json"])
            if row.get("aliases", "").strip():
                case["aliases"] = [a.strip() for a in row["aliases"].split("||") if a.strip()]
            for k in ("language", "doc_lang", "doc_type", "qtype"):
                if row.get(k, "").strip():
                    case[k] = row[k].strip()
            out.append(case)
    with open(dst, "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return len(out)


if __name__ == "__main__":
    print(convert(sys.argv[1], sys.argv[2]), "cases written")
