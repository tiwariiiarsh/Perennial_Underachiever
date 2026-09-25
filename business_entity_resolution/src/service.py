"""Inference service shared by the FastAPI backend (and usable from notebooks).

It reuses exactly the pipeline functions (normalize_df, PoolIndex, pair_features,
decide) so API answers are identical to the batch pipeline for the same pool.
The searchable pool defaults to the first ER_API_POOL_ROWS records of the test
Source-2/3 pool to keep start-up fast on a laptop.
"""
import json
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from blocking import KEY_TYPES, PoolIndex, build_keys, unpack
from config import MODEL_DIR, TOP_K
from decide import decide
from features import FEAT_NAMES, pair_features
from normalize import normalize_df
from prepare import get_norm

EXPLAIN = ["n_tset", "n_jw", "k_jac", "kc_ratio", "suffix_agree", "num_first_eq", "num_jac",
           "a_tset", "aw_jac", "country_eq", "blk_score"]


def _records(rows):
    df = pd.DataFrame(rows)
    for c in ("business_name", "business_address", "country"):
        df[c] = df.get(c, "").fillna("").astype(str) if c in df else ""
    if "entity_id" not in df:
        df["entity_id"] = [f"Q-{i}" for i in range(len(df))]
    return df


class Matcher:
    def __init__(self, pool_rows=None):
        self.booster = lgb.Booster(model_file=str(MODEL_DIR / "lgb.txt"))
        self.iso = pickle.load(open(MODEL_DIR / "iso.pkl", "rb"))
        self.params = json.load(open(MODEL_DIR / "decision.json"))
        self.metrics = json.load(open(MODEL_DIR / "metrics.json"))
        pool_rows = int(os.environ.get("ER_API_POOL_ROWS", pool_rows or 1_000_000))
        _, pool = get_norm("test")
        self.pool = pool.iloc[:pool_rows].reset_index(drop=True)
        self.index = PoolIndex(self.pool)

    # ------------------------------------------------------------------ core
    def _score(self, q_norm, cands, pool_norm):
        X = pair_features(cands, q_norm, pool_norm)
        cands = cands.copy()
        cands["prob"] = self.iso.predict(self.booster.predict(X[self.booster.feature_name()]))
        return cands, X

    def match(self, rows, top_k=TOP_K):
        """rows: list of {business_name, business_address, country}.  Returns per-row results."""
        q = normalize_df(_records(rows), n_workers=1)
        cands = self.index.query(q, top_k)
        out = [{"query": r, "candidates": [], "matches": []} for r in rows]
        if cands.empty:
            return out
        cands, X = self._score(q, cands, self.pool)
        chosen = set(zip(*decide(cands, self.params)[["s1_idx", "pool_idx"]].to_numpy().T))
        raw = self.pool
        for (i, row), (_, xr) in zip(cands.sort_values("prob", ascending=False).iterrows(),
                                     X.loc[cands.sort_values("prob", ascending=False).index].iterrows()):
            s, p = int(row.s1_idx), int(row.pool_idx)
            item = {"entity_id": raw.entity_id[p], "name": raw.name[p], "address": raw.addr[p],
                    "country": raw.country[p], "probability": round(float(row.prob), 4),
                    "matched": (s, p) in chosen,
                    "evidence": {k: round(float(xr[k]), 3) for k in EXPLAIN}}
            out[s]["candidates"].append(item)
            if item["matched"]:
                out[s]["matches"].append(item["entity_id"])
        return out

    def compare(self, a, b):
        """Probability that two raw records are the same business + feature breakdown."""
        qa, qb = normalize_df(_records([a]), 1), normalize_df(_records([b]), 1)
        ha, ta, _ = unpack(build_keys(qa))
        hb, _, _ = unpack(build_keys(qb))
        shared = np.isin(ha, hb)
        pos = np.clip(np.searchsorted(self.index.ukeys, ha), 0, len(self.index.ukeys) - 1)
        known = self.index.ukeys[pos] == ha
        w = np.where(known, self.index.weight[pos], np.log(self.index.n)).astype(np.float32) * shared
        cands = pd.DataFrame({"s1_idx": [0], "pool_idx": [0], "blk_score": [w.sum()],
                              "blk_nkeys": [int(shared.sum())], "blk_rank": [0]})
        for ti, name in enumerate(KEY_TYPES):
            cands["blk_" + name] = float(w[ta == ti].sum())
        cands, X = self._score(qa, cands, qb)
        prob = float(cands.prob.iloc[0])
        return {"probability": round(prob, 4), "match": prob >= self.params["min_prob"],
                "normalized_a": qa.iloc[0].to_dict(), "normalized_b": qb.iloc[0].to_dict(),
                "features": {k: round(float(X[k].iloc[0]), 3) for k in FEAT_NAMES + ["blk_score"]}}

    def batch(self, df):
        """df in Source-1 format -> DataFrame(source1_entity_id, matched_entity_ids)."""
        res = self.match(df[["business_name", "business_address", "country"]].to_dict("records"))
        return pd.DataFrame({"source1_entity_id": df.entity_id.to_numpy(),
                             "matched_entity_ids": [",".join(r["matches"]) for r in res]})
