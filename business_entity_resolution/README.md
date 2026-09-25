# Business Entity Resolution — reproducible pipeline

Blocking (weighted inverted index) → pairwise features → LightGBM + isotonic calibration →
one-owner rule + expected-F0.5 subset selection → `output/matching_results.tsv` and
`output/candidate_pairs.tsv`.  A FastAPI backend and a Streamlit UI sit on top for demos.

## 1. Setup

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r business_entity_resolution/requirements.txt
```

Data layout expected (override with `ER_DATA_DIR=/path/to/dataset`):

```
student_resource/dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
student_resource/dataset/test/{test_source1,test_source2,test_source3}.tsv
```

## 2. Reproduce the submission (one command)

```bash
cd business_entity_resolution
python src/run_pipeline.py            # trains if models/ is empty, then predicts test
python ../student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir ../student_resource/dataset/test
```

Stages (each can also be run alone from `src/`):

| Step | Command | Output |
|---|---|---|
| Normalise + cache | `python src/prepare.py train test` | `work/*_norm.parquet` |
| Blocking recall report | `python src/eval_blocking.py --k 20 30 40` | recall table + missed pairs |
| Train + validate | `python src/train.py` | `models/lgb.txt`, `iso.pkl`, `decision.json`, `metrics.json` |
| Predict test | `python src/run_pipeline.py` | `output/*.tsv` |
| Build zip | `python src/make_submission.py --team <name>` | `<name>_submission.zip` |

Everything is seeded (`SEED=42` in `src/config.py`); outputs are deterministic for a given machine.

## 3. Demo app

```bash
cd business_entity_resolution
uvicorn api.main:app --port 8000          # backend (indexes first 1M test pool rows; ER_API_POOL_ROWS to change)
streamlit run app/streamlit_app.py        # frontend at http://localhost:8501
```

API endpoints: `GET /health`, `POST /match`, `POST /compare`, `POST /batch` (TSV upload), `GET /stats`.

## 4. Runtime / hardware

Developed on a MacBook (Apple Silicon, 8 CPU cores, 8 GB RAM), CPU only, no GPU.
Normalisation ≈10 min, training ≈30–40 min, test inference ≈30 min. Peak RAM ≈3–5 GB.

## 5. Code map

```
src/config.py        paths, K, thresholds, LightGBM params, seed
src/io_utils.py      TSV read/write (always sep="\t")
src/normalize.py     transliteration, legal suffixes, aliases (DBA/FKA), consonant skeletons, address parsing
src/prepare.py       normalise + parquet cache
src/blocking.py      hashed-key inverted index, idf-weighted top-K candidates
src/features.py      name / address / blocking-context pair features (rapidfuzz)
src/train.py         GroupKFold LightGBM, isotonic calibration, decision-param grid search, metrics.json
src/decide.py        one-owner rule + expected-F0.5 selection + probability floor
src/metric.py        exact macro F0.5 (singletons included)
src/run_pipeline.py  end-to-end entry point
src/service.py       shared inference layer for the API
api/main.py          FastAPI backend
app/streamlit_app.py Streamlit frontend
```

No external data, APIs, geocoders or pretrained models are used — only the provided training files.
All dependencies are permissively licensed (BSD/MIT/Apache-2.0); LightGBM is MIT.
