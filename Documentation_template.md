# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Perennial_Underachiever
**Team Members:** Arsh Tiwari
**Submission Date:** 2026-09-25

---

## 1. Executive Summary

We resolve Source-2/3 records to deduplicated Source-1 entities with a classic but carefully engineered
*block → score → decide* pipeline that runs on an 8 GB laptop over ~12 M records per split. The key ideas are
(i) script- and typo-robust **consonant-skeleton keys** (so Devanagari/Telugu/Kannada/Bengali names, typos and
"leet" digits collide with their Latin forms), (ii) an **idf-weighted inverted-index blocker** that mixes name,
address and name×address keys, and (iii) a **decision layer** tailored to macro F0.5: a one-owner constraint plus
per-entity *expected-F0.5* subset selection (which may return the empty list), on top of a calibrated LightGBM.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the training set (2,206,821 Source-1 records; 10,320,219 Source-2/3 records; US 60% / India 40%):

| Statistic | Value |
|---|---|
| Source-1 singletons (no match) | 123,247 (5.6%) |
| Mean matches per Source-1 entity | 3.46 (mode 3; up to 8+) |
| Matched records from S2 / S3 | 3.69 M / 3.94 M |
| Pool records matched to >1 Source-1 entity | **0** → one-owner rule is exact |
| Pool records matched to no Source-1 entity | ~26% (distractors) |
| Test Source-1 countries | India 810k, US 663k, **France 259k (unseen)** |

Observed noise patterns (manual inspection of ~100 matched clusters):

* **Names** – legal suffix drift (`Pvt`/`Private`/`(Ltd)`/`[Incorporated]`/`P.C.`/`S.A.S`), honorifics (`Mr`,
  `M/s`), stutters (`Renaissancere Renaissancere`, `Bailey, Bailey`), typos (`Makreitrdng`, `Dnegtla`), leet/OCR
  digits (`Tayl0r`, `C1inic`, `0rthopedic`), word-order transpositions, noise prefixes (`--`, `<<`), bracketed
  descriptors (`[Services]`), **native-script names** (Devanagari, Telugu, Kannada, Bengali), **website domains**
  (`wilkdeltaplus.com`), alias constructions (`X doing business as Y`, `X formerly known as Y`, `F/K/A`) and
  occasionally a completely different trade name with an identical address.
* **Addresses** – abbreviations (`St/Street`, `Dr/Drive`, `R./Rue`, `AV/Avenue`), component reordering
  (`MA, Falmouth, 4 Sandy Lane`), missing house numbers / PIN / state, state names vs codes (`TG`/`Telangana`,
  `పశ్చిమబంగ`), mangled numbering (`12-11-1178`, `1-6/7`, `C-##59`, `4845-`), house-number typos (`3385` vs `385`),
  landmarks and fully empty addresses.

### 2.2 Solution Strategy

**Approach Type:** Blocking + gradient-boosted pair classifier + F0.5-aware set decision (hybrid, constraint-based).
**Core Innovation:** Transliteration-invariant consonant-skeleton keys in an idf-weighted inverted index, and a
decision layer that directly maximises expected macro F0.5 per entity under a one-owner constraint.

Pipeline: `normalize → block (top-K) → pair features → LightGBM (GroupKFold OOF) → isotonic calibration →
one-owner → expected-F0.5 selection → TSVs`.

Validation protocol: a random sample of 200,000 training Source-1 entities is blocked against the **full** training
pool (exactly the test setting, including distractors). 5-fold GroupKFold by Source-1 id yields out-of-fold
probabilities; calibration and all decision parameters are tuned on those OOF predictions with the exact macro F0.5
metric (`src/metric.py`, verified on the PS example = 0.714).

---

## 3. Candidate Generation (Blocking)

**Normalisation** (`src/normalize.py`, identical for every country):
unidecode transliteration → lowercase → `&`→`and` → domain stripping (`x.com`→`x`) → dotted-initials merge
(`p.c.`→`pc`) → leet digit repair inside words → legal-suffix extraction (canonical `inc/corp/co/llc/ltd/pvt/pc/llp/
sarl/sas/sci/eurl/…`, also recognised through skeletons such as `praaivett`→`prbt`) → leading honorific removal →
stutter de-duplication → alias split (text after DBA/FKA/AKA/"anciennement").
Each token also gets a **consonant skeleton**: `ph→f, ck/c/q→k, x→ks, v/w→b, z→s, j→g`, aspirate `h` dropped,
vowels dropped, repeats collapsed (`maarkettiNg`, `Marketing` → `mrktng`; `inbhesttmentt`, `investment` →
`nbstmnt`). Addresses: comma segments, abbreviation expansion (US + Indian + French street types), numbers
canonicalised by joining digit groups (`16/7`, `1-6/7` → `167`), generic words removed for keys.

**Blocking keys used** (hashed to uint64, `src/blocking.py`):

| Key | Example | Type |
|---|---|---|
| name-token skeleton | `n:mrktng` | name |
| concatenated name skeleton (domains, spacing) | `c:blkdltpls` | name |
| sorted pairs of name tokens | `p:lksm mrktng` | name |
| address word bigram / unigram skeletons | `a:snd ln`, `u:hlh` | address |
| house number × street word | `h:4474 hlh` | address |
| name token × house number | `nh:lksm 12111178` | mixed |
| name token × address word (order-free) | `nw:tlr pn` | mixed |

Keys shared by more than 300 pool records are discarded; every remaining key contributes `idf = log(N/df)` to the
pair score, split into name/address/mixed sub-scores. The top **K = 30** pool records per Source-1 entity are
kept. Everything is numpy (sorted keys + `searchsorted` + `bincount`), which indexes 10 M records in 8 GB RAM.

**Blocking quality (50,000 training S1 entities vs. full pool):**

| Version | K | Candidates / S1 | Pair recall |
|---|---|---|---|
| v1 (name + address keys, max_df 60) | 20 | 18.8 | 0.899 |
| v1 | 40 | 34.3 | 0.918 |
| v2 (+ leet repair, name×address-word keys, max_df 300) | 20 | 19.9 | **0.936** |
| v2 | 40 | 39.4 | **0.947** |

- **Candidate pairs generated:** 5,889,320 for the 200k-entity validation sample (29.4 per Source-1 entity, pair recall **0.939**); for test ≈30 per Source-1 entity (see `output/candidate_pairs.tsv`). Reduction ratio vs. the full cross product ≈ 1 − 30/10⁷ ≈ 99.9997%.
- **How true matches were not lost:** missed-pair analysis after every blocking change (`src/eval_blocking.py
  --show-misses`) drove each key type: skeletons for transliteration/typos, concatenated skeletons for domain
  names, order-free name×address-word keys for reordered addresses without house numbers, and a high df cap with
  idf weighting so common names are still retrievable when combined with an address. Remaining misses are mostly
  records with a common name **and** an empty address (e.g. `Heritage Foundation` with no address), which the
  precision-oriented metric would not reward matching anyway.

---

## 4. Matching Model

**Features used** (`src/features.py`, 36 features). Top features by gain: total blocking idf score, candidate rank, name×address key score, house-number Jaccard, address token-set ratio, concatenated-skeleton ratio, pool-side number count, name Jaro-Winkler, partial ratio, legal-suffix agreement.
- **Name:** rapidfuzz token-set / token-sort / plain / partial ratios and Jaro-Winkler on the transliterated core
  name (max over main name and alias), same ratios on skeleton strings and concatenated skeletons, skeleton-token
  Jaccard and overlap count, token counts, first-token equality, legal-suffix agreement (+1 agree / −1 conflict /
  0 unknown), length difference, native-script flag, alias present.
- **Address:** house-number Jaccard and first-number equality, number counts, address token-set ratio,
  address-word-skeleton Jaccard and overlap, missing-address flag.
- **Blocking context:** total / name / address / mixed idf scores, number of shared keys, rank within the S1
  candidate list, ratio and gap to the best candidate, candidate count, name score relative to best candidate.
- **Other:** source (S2/S3), country *equality* flag only (country is never one-hot encoded, so France is handled
  by the same model).

**Model type:** LightGBM binary classifier (MIT licence; ~600 trees, 127 leaves), 5-fold GroupKFold by Source-1 id
for out-of-fold predictions, refit on all pairs for test; isotonic regression calibration on OOF scores.
No pretrained language model is used (well under the 8B-parameter limit).

**Threshold selection method:** direct grid search of the decision parameters on OOF macro F0.5:
1. *One-owner rule:* each pool record is assigned only to its highest-probability Source-1 entity (optionally
   requiring a margin over the runner-up) — exact in training GT.
2. *Expected-F0.5 selection:* per entity, sort candidates by probability and choose the prefix size k maximising
   `E[F0.5](k) ≈ 1.25·Σ_{i≤k} p_i / (0.25·(Σp + miss) + k)`, versus `E[F0.5](0) = Π(1−p_i)` for the empty list.
3. *Probability floor:* never predict pairs with calibrated p below `min_prob`.

Chosen parameters (grid search on OOF macro F0.5): `margin = 0.1`, `min_prob = 0.6`, `miss = 0.3`.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, OOF validation on 200k S1 entities):** **0.9383**
- **Per country:** US 0.9476 · India 0.9242
- **Pair-level ROC-AUC / PR-AUC:** 0.99935 / 0.99505 (5 folds, AUC 0.99931–0.99936)

**Decision-logic ablation (OOF macro F0.5):**

| Configuration | Macro F0.5 |
|---|---|
| Predict nothing (floor = singleton rate) | 0.0562 |
| Calibrated p ≥ 0.5 | 0.9319 |
| Expected-F0.5 selection + probability floor, no one-owner | 0.9379 |
| + one-owner rule with margin (final) | **0.9383** |

- **Common false positives (wrong merges):** sibling businesses sharing a building and a generic name token
  (`Blue Foundation` vs `Blue Software` at the same address), chains of the same brand in one city, and records
  whose only evidence is a common name with no address.
- **Common false negatives (missed matches):** name-only records with generic names and no address, records whose
  name is a completely different trade name *and* whose address lost its house number, and heavily corrupted
  native-script transliterations.

**France strategy (unseen country):** no feature or rule depends on the country value; normalisation includes
French street types (`r.`→rue, `av`→avenue, `bd`→boulevard, `all`→allée, `imp`, `rte`, `ch`, `fbg`), French legal
forms (`SARL, SAS, SASU, SA, EURL, SCI, SNC, SCP, SELARL, Cie`) and accent folding, so French records reach the
same feature space as training data.

---

## 6. Conclusion

A lean, fully CPU-based pipeline shows that careful normalisation (transliteration + consonant skeletons),
idf-weighted multi-key blocking, and a metric-aware decision layer (one-owner + expected F0.5) are the main drivers
for this task. The biggest lessons: measure blocking recall on missed pairs before modelling, and optimise the
set-decision for the actual macro, singleton-inclusive F0.5 rather than a pair-level threshold.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

```
src/config.py  io_utils.py  normalize.py  prepare.py  blocking.py  features.py
src/train.py   decide.py    metric.py     run_pipeline.py  eval_blocking.py  service.py  make_submission.py
api/main.py            FastAPI backend (/health /match /compare /batch /stats)
app/streamlit_app.py   Streamlit demo UI
README.md  requirements.txt
```

Entry point: `python src/run_pipeline.py` (normalise → train if needed → block/score/decide test → write
`output/matching_results.tsv` and `output/candidate_pairs.tsv`).

### B. Training run

Training (200k S1 entities, 5.9M pairs) took 24 min wall-clock on 8 CPU cores with 4.0 GB peak RAM; LightGBM folds took ~80 s each.

### C. Fair play / licences

No external databases, APIs, geocoders, web lookups or pretrained models were used; all signal comes from the
provided training files. Libraries: pandas, numpy, pyarrow, scikit-learn (BSD), LightGBM (MIT), rapidfuzz (MIT),
Unidecode (GPL-2.0, used only as a local transliteration table), FastAPI/Streamlit for the demo (MIT/Apache-2.0).
