"""One command: data -> normalisation -> blocking -> features -> model -> decision -> TSVs.

    python src/run_pipeline.py            # trains if models/ is empty, then predicts test
    python src/run_pipeline.py --retrain  # force retraining
"""
import argparse
import json
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from blocking import PoolIndex
from config import MODEL_DIR, OUTPUT_DIR, TOP_K
from decide import decide
from features import pair_features
from io_utils import write_id_lists
from prepare import get_norm


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def predict_test(chunk=150_000):
    booster = lgb.Booster(model_file=str(MODEL_DIR / "lgb.txt"))
    iso = pickle.load(open(MODEL_DIR / "iso.pkl", "rb"))
    params = json.load(open(MODEL_DIR / "decision.json"))
    s1, pool = get_norm("test")
    log("test", s1.shape, pool.shape, "building pool index")
    index = PoolIndex(pool)
    parts = []
    for start in range(0, len(s1), chunk):
        sub = s1.iloc[start:start + chunk].reset_index(drop=True)
        c = index.query(sub, TOP_K)
        X = pair_features(c, sub, pool)
        c = c[["s1_idx", "pool_idx"]].copy()
        c["prob"] = iso.predict(booster.predict(X[booster.feature_name()]))
        c["s1_idx"] += start
        parts.append(c)
        log(f"scored {min(start + chunk, len(s1))}/{len(s1)}")
    del index
    cands = pd.concat(parts, ignore_index=True)
    chosen = decide(cands, params)            # one-owner is applied across ALL test S1
    s1_ids, pid = s1.entity_id.to_numpy(), pool.entity_id.to_numpy()

    def to_map(df):
        m = {}
        for s, p in zip(s1_ids[df.s1_idx.to_numpy()], pid[df.pool_idx.to_numpy()]):
            m.setdefault(s, []).append(p)
        return m

    write_id_lists(OUTPUT_DIR / "candidate_pairs.tsv", s1_ids, to_map(cands), "candidate_entity_ids")
    write_id_lists(OUTPUT_DIR / "matching_results.tsv", s1_ids, to_map(chosen), "matched_entity_ids")
    n_non_empty = chosen.s1_idx.nunique()
    log(f"wrote outputs: candidates/s1={len(cands)/len(s1):.1f} matches/s1={len(chosen)/len(s1):.2f} "
        f"non-empty={n_non_empty/len(s1):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    if a.retrain or not (MODEL_DIR / "lgb.txt").exists():
        import train
        train.main()
    predict_test()
