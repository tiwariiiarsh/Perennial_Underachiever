"""Pairwise features for (Source-1, candidate) pairs.  Country is only used as
an equality flag, never as a category, so unseen countries behave normally."""
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from config import N_WORKERS

TEXT_COLS = ["name", "alt", "suffix", "name_skel", "nums", "addr", "addr_skel", "country"]


def _jac(a, b):
    if not a or not b:
        return -1.0
    return len(a & b) / len(a | b)


def _pair(r1, r2):
    n1, a1, suf1, k1, num1, ad1, ak1, c1 = r1
    n2, a2, suf2, k2, num2, ad2, ak2, c2 = r2
    kt1, kt2 = set(k1.split()), set(k2.split())
    names2 = [x for x in (n2, a2) if x]
    tsr = max((fuzz.token_set_ratio(n1, x) for x in names2), default=0)
    tso = max((fuzz.token_sort_ratio(n1, x) for x in names2), default=0)
    jw = max((JaroWinkler.similarity(n1, x) for x in names2), default=0)
    pr = max((fuzz.partial_ratio(n1, x) for x in names2), default=0)
    k1c, k2c = k1.replace(" ", ""), k2.replace(" ", "")
    s1s, s2s = set(suf1.split()), set(suf2.split())
    nu1, nu2 = num1.split(), num2.split()
    ns1, ns2 = set(nu1), set(nu2)
    w1 = set(ak1.replace("|", " ").split())
    w2 = set(ak2.replace("|", " ").split())
    return (
        tsr, tso, fuzz.ratio(n1, n2), jw, pr,
        fuzz.ratio(k1, k2), fuzz.token_set_ratio(k1, k2), fuzz.ratio(k1c, k2c), fuzz.partial_ratio(k1c, k2c),
        _jac(kt1, kt2), len(kt1 & kt2), len(kt1), len(kt2), int(bool(a2)),
        int(k1.split()[:1] == k2.split()[:1]) if k1 and k2 else -1,
        (1 if s1s & s2s else -1 if (s1s and s2s) else 0),
        _jac(ns1, ns2), int(nu1[:1] == nu2[:1]) if nu1 and nu2 else -1,
        len(nu1), len(nu2),
        fuzz.token_set_ratio(ad1, ad2) if ad1 and ad2 else -1,
        _jac(w1, w2), len(w1 & w2), int(not ad2), len(n2), abs(len(n1) - len(n2)),
        int(c1 == c2),
    )


FEAT_NAMES = [
    "n_tset", "n_tsort", "n_ratio", "n_jw", "n_partial",
    "k_ratio", "k_tset", "kc_ratio", "kc_partial",
    "k_jac", "k_common", "k_len1", "k_len2", "has_alias", "k_first_eq",
    "suffix_agree", "num_jac", "num_first_eq", "num_len1", "num_len2",
    "a_tset", "aw_jac", "aw_common", "a_missing2", "n_len2", "n_lendiff", "country_eq",
]


def _feat_chunk(args):
    a, b = args
    return np.array([_pair(x, y) for x, y in zip(a, b)], dtype=np.float32)


def pair_features(cands, s1_norm, pool_norm, chunk=50_000):
    """cands: DataFrame with s1_idx, pool_idx, blk_* columns.  Returns feature frame."""
    left = list(zip(*[s1_norm[c].to_numpy()[cands.s1_idx.to_numpy()] for c in TEXT_COLS]))
    right = list(zip(*[pool_norm[c].to_numpy()[cands.pool_idx.to_numpy()] for c in TEXT_COLS]))
    jobs = [(left[i:i + chunk], right[i:i + chunk]) for i in range(0, len(left), chunk)]
    if not jobs:
        arr = np.zeros((0, len(FEAT_NAMES)), np.float32)
    elif len(left) < 20_000:                    # small (API) requests: no process pool
        arr = np.vstack([_feat_chunk(j) for j in jobs])
    else:
        with Pool(N_WORKERS) as p:
            arr = np.vstack(p.map(_feat_chunk, jobs))
    del left, right
    f = pd.DataFrame(arr, columns=FEAT_NAMES, index=cands.index)
    for c in ["blk_score", "blk_name", "blk_addr", "blk_mixed", "blk_nkeys", "blk_rank"]:
        f[c] = cands[c].to_numpy()
    g = cands.groupby("s1_idx").blk_score
    f["blk_rel"] = cands.blk_score / g.transform("max")
    f["blk_gap"] = g.transform("max") - cands.blk_score
    f["blk_ncand"] = g.transform("size")
    f["is_s3"] = pool_norm.entity_id.str.startswith("S3").to_numpy()[cands.pool_idx.to_numpy()].astype(np.int8)
    f["nonlatin2"] = pool_norm.nonlatin_name.to_numpy()[cands.pool_idx.to_numpy()]
    # within-S1 relative name score
    f["n_tset_rel"] = f.n_tset - f.groupby(cands.s1_idx).n_tset.transform("max")
    return f
