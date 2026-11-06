# Test cases and evaluation

The assignment asks for **30–50 representative test cases**. `testcases.jsonl` ships with **50 draft cases
(48 QA + 2 extraction) over 4 public documents** in `data/samples/`:

| Document | Language | Cases | Notes |
|---|---|---|---|
| Gazette of India notification (land acquisition, 13 pp) | Hindi | 20 | tables across pages; garbled text layer; 2 English questions |
| AIR Bengaluru regional bulletin (3 pp) | Kannada | 10 | garbled text layer; 1 English question |
| Maharashtra police circular + Finance Dept GR (6 pp) | Marathi | 8 | first 2 pages are camera scans (no text layer) |
| AIR Hyderabad regional bulletin (11 pp, Telugu then English) | Telugu | 12 | prices / counts written as digits plus Telugu words (e.g. "2 వేల 78 రూపాయల") |

All labels are **drafts** (`"label_status": "draft"`): check each against the page image and set it to
`verified`. Four documents and no Tamil/Bengali/forms means the set is **not yet fully representative**:
add the missing types below before you report numbers.

## Recommended coverage (aim for ~40 cases)

| Dimension | Target |
|---|---|
| Documents | 8–12 different documents |
| Languages | Hindi plus at least 2 others (e.g. Kannada, Marathi, Tamil, Bengali) |
| Document types | gazette notification, government form, utility/tax notice, certificate, circular, table-heavy |
| Question types (`qtype`) | `factoid`, `date`, `identifier`, `numeric`, `table_lookup`, `multi_field`, `span`, `cross_lingual` |
| Question language | native-language questions **and** English questions on native-language documents |
| Pages | include answers on page > 1 so retrieval is actually tested |

## Case format (one JSON object per line)

```json
{"id": "q-011", "kind": "qa", "doc": "data/samples/x.pdf", "page": 1,
 "question": "...", "answer": "0.2095", "aliases": ["0.2095 हेक्टेयर"],
 "language": "hi", "doc_lang": "hi", "doc_type": "gazette_notification", "qtype": "table_lookup"}

{"id": "ex-001", "kind": "extraction", "doc": "...", "page": 0,
 "fields": {"date": ["27 अक्टूबर, 2026", "27/10/2026"], "state": "गुजरात", "taluk": null},
 "language": "hi", "doc_lang": "hi", "doc_type": "...", "qtype": "field_extraction"}
```

- `page` is **0-based** and is the page that contains the answer (used for the retrieval hit-rate).
- `aliases` / list values are additional accepted answers (date formats, spelling variants).
- A `null` field means "not present"; it is not scored.
- An extraction case is scored per field, so `ex-001` contributes 7 rows to the metrics.

## Labelling rules (learned the hard way)

1. **Name the context in table questions.** Survey / row numbers repeat across villages or sections
   (survey 108 occurs in three villages in the sample). Always put the village/section in the question.
2. **Label from the page image, not from the PDF text layer.** Gazette PDFs often use legacy fonts, so the
   embedded text decodes to wrong Devanagari.
3. Keep gold answers **short and copied exactly** as printed.
4. Don't label documents with personal data (Aadhaar, PAN, phone numbers); use public documents.

## Labelling in a spreadsheet

Fill a Google Sheet / CSV with the columns
`id,kind,doc,page,question,answer,aliases,fields_json,language,doc_lang,doc_type,qtype`
(`aliases` separated by ` || `), export as CSV, then:

```bash
python eval/csv_to_jsonl.py my_cases.csv eval/testcases.jsonl
```

## Running

```bash
python eval/run_eval.py --out eval/results/run1               # full pipeline: retrieval + answer + verify
python eval/run_eval.py --out eval/results/run1_goldpage --gold-page   # VLM only (skips retrieval)
python eval/run_eval.py --out eval/results/run1_noverify --no-verify   # ablation: no self-verification
```

Run all three: the differences between them are the ablation table for your report
(retrieval error vs reading error vs verification effect).

Outputs per run: `results.jsonl` (one row per scored item), `summary.json`
(overall + breakdowns + bootstrap 95% CI on EM + latency), and `failure_analysis.md`
(error taxonomy with worst examples per tag and a section for your manual analysis).

## Metrics

- **EM**: exact match after normalisation (NFC, ASCII digits, punctuation/zero-width removed, lower-case).
- **ANLS**: DocVQA's Average Normalised Levenshtein Similarity (threshold 0.5): partial credit for
  small spelling / matra differences.
- **token-F1**: bag-of-words overlap.
- **Retrieval hit-rate**: gold page ∈ pages retrieved (needs `page` in the case).
- Field extraction accuracy = the same metrics averaged over scored fields.
