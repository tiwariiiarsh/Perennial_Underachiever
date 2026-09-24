"""Turn calibrated pair probabilities into per-Source-1 match lists.

1. One-owner rule: a Source-2/3 record belongs to at most one Source-1 entity
   (never violated in the training ground truth), so each pool record is kept
   only for its highest-probability Source-1 owner and only if it beats the
   runner-up by `margin`.
2. Expected-F0.5 selection: candidates of one Source-1 entity are sorted by
   probability and the prefix size k (0 = predict empty) that maximises
       E[F0.5](k) ~= 1.25 * sum_{i<=k} p_i / (0.25 * T + k),  T = sum p_i + miss,
       E[F0.5](0) ~= prod (1 - p_i)
   is chosen.
3. Safety floor: pairs below `min_prob` are never predicted.
"""
import numpy as np
import pandas as pd

DEFAULT_PARAMS = dict(margin=0.0, min_prob=0.3, miss=0.05)


def one_owner(df, margin):
    """df: s1_idx, pool_idx, prob. Keeps best owner per pool record."""
    df = df.sort_values(["pool_idx", "prob"], ascending=[True, False], kind="stable")
    rank = df.groupby("pool_idx").cumcount()
    second = df.groupby("pool_idx").prob.shift(-1)
    keep = (rank == 0) & ((second.isna()) | (df.prob - second.fillna(0) >= margin))
    return df[keep.to_numpy()]


def select(df, min_prob, miss):
    """Expected-F0.5 prefix selection per s1_idx. Returns the selected rows."""
    df = df.sort_values(["s1_idx", "prob"], ascending=[True, False], kind="stable")
    p = df.prob.to_numpy()
    g = df.s1_idx.to_numpy()
    grp = pd.Series(p).groupby(g)
    k = grp.cumcount().to_numpy() + 1
    csum = grp.cumsum().to_numpy()
    tot = grp.transform("sum").to_numpy() + miss
    ef = 1.25 * csum / (0.25 * tot + k)
    log_none = pd.Series(np.log1p(-np.clip(p, 0, 1 - 1e-6))).groupby(g).transform("sum").to_numpy()
    ef0 = np.exp(log_none)
    best = pd.Series(ef).groupby(g).transform("max").to_numpy()
    # k* = first prefix reaching the max; keep rows with rank <= k*
    is_best = ef >= best - 1e-12
    kstar = pd.Series(np.where(is_best, k, 1 << 30)).groupby(g).transform("min").to_numpy()
    keep = (k <= kstar) & (best > ef0) & (p >= min_prob)
    return df[keep]


def decide(df, params=None):
    params = {**DEFAULT_PARAMS, **(params or {})}
    owned = one_owner(df[["s1_idx", "pool_idx", "prob"]], params["margin"])
    return select(owned, params["min_prob"], params["miss"])
