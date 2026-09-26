"""Candidate generation: weighted inverted-index blocking.

Every record emits a handful of string keys (name-token skeletons, name-token
pairs, the concatenated name skeleton for domain-style names, address word
bigrams, house-number + street-word, name + house-number).  Keys are hashed to
uint64 and indexed for the Source-2/3 pool.  Keys shared by more than
KEY_MAX_DF pool records are discarded (too common to be informative); every
other key contributes idf = log(N / df) to the (s1, pool) pair score.  The
TOP_K highest-scoring pool records per Source-1 record are the candidates.
"""
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd

from config import KEY_MAX_DF, TOP_K, N_WORKERS
from normalize import skel

KEY_TYPES = ("name", "addr", "mixed")


def record_keys(name_skel, alt, nums, addr_skel):
    """Return list of (key, type) with type index into KEY_TYPES."""
    keys = []
    toks = [t for t in dict.fromkeys(name_skel.split()) if len(t) >= 2]
    alt_toks = [t for t in (skel(a) for a in alt.split()) if len(t) >= 2]
    for t in toks + alt_toks:
        keys.append(("n:" + t, 0))
    for tl in (toks, alt_toks):
        c = "".join(tl)
        if len(c) >= 5:
            keys.append(("c:" + c, 0))
    for a, b in combinations(sorted(toks[:4]), 2):
        keys.append(("p:" + a + " " + b, 0))
    words = []
    for seg in addr_skel.split("|") if addr_skel else ():
        sw = [w for w in seg.split() if len(w) >= 2]
        words.extend(sw)
        for a, b in zip(sw, sw[1:]):
            keys.append(("a:" + a + " " + b, 1))
    words = [w for w in dict.fromkeys(words) if len(w) >= 3]
    num_list = list(dict.fromkeys(nums.split()))[:4]
    for n in num_list[:2]:
        for w in words[:3]:
            keys.append(("h:" + n + " " + w, 1))
    if toks and num_list:
        keys.append(("nh:" + toks[0] + " " + num_list[0], 2))
        if len(toks) > 1:
            keys.append(("nh:" + toks[1] + " " + num_list[0], 2))
    # Postal code + name anchor (5 digits for France/US, 6 digits for India)
    for n in num_list:
        if len(n) in (5, 6) and toks:
            keys.append(("z:" + n + " " + toks[0], 2))
            break
    for t in toks[:2]:
        for w in words[:4]:
            keys.append(("nw:" + t + " " + w, 2))
    return list(dict(keys).items())


# Every (key, type, record) posting is packed into one uint64:
#   bits 26..63 = 38-bit key hash, bits 24..25 = key type, bits 0..23 = record index (< 16.7M)
# so the index is a single in-place-sorted array (no argsort copies) -> fits 10M+ records in 8 GB.
REC_BITS, KEY_SHIFT = 24, 26
REC_MASK = (1 << REC_BITS) - 1


def _keys_chunk(args):
    offset, cols = args
    ks, idx, typ = [], [], []
    for i, row in enumerate(zip(*cols)):
        for k, t in record_keys(*row):
            ks.append(k)
            idx.append(offset + i)
            typ.append(t)
    if not ks:
        return np.zeros(0, np.uint64)
    h = pd.util.hash_array(np.array(ks, dtype=object)) >> np.uint64(KEY_SHIFT) << np.uint64(KEY_SHIFT)
    return h | (np.array(typ, np.uint64) << np.uint64(REC_BITS)) | np.array(idx, np.uint64)


def build_keys(norm, chunk=100_000):
    """Packed uint64 postings for every record of `norm` (see bit layout above)."""
    assert len(norm) <= REC_MASK, "record index does not fit in 24 bits; query in smaller chunks"
    cols = [norm[c].tolist() for c in ("name_skel", "alt", "nums", "addr_skel")]
    jobs = [(i, [c[i:i + chunk] for c in cols]) for i in range(0, len(norm), chunk)]
    if len(norm) < 20_000:                      # small (API) requests: no process pool
        parts = [_keys_chunk(j) for j in jobs]
    else:
        with Pool(N_WORKERS) as p:
            parts = p.map(_keys_chunk, jobs)
    del cols, jobs
    return np.concatenate(parts) if parts else np.zeros(0, np.uint64)


def unpack(packed):
    return (packed >> np.uint64(KEY_SHIFT)), ((packed >> np.uint64(REC_BITS)) & np.uint64(3)).astype(np.int8), \
        (packed & np.uint64(REC_MASK)).astype(np.int32)


class PoolIndex:
    def __init__(self, pool_norm, max_df=KEY_MAX_DF, step=20_000_000):
        arr = build_keys(pool_norm)
        arr.sort()                                            # in place
        n = len(arr)
        # posting-list boundaries, computed in slices to bound temporary memory
        bounds = [np.zeros(1, np.int64)]
        for a in range(1, n, step):
            b = min(a + step, n)
            k1 = arr[a:b] >> np.uint64(KEY_SHIFT)
            k0 = arr[a - 1:b - 1] >> np.uint64(KEY_SHIFT)
            bounds.append(np.flatnonzero(k1 != k0).astype(np.int64) + a)
            del k0, k1
        starts = np.concatenate(bounds)
        counts = np.diff(np.append(starts, n))
        keep = counts <= max_df
        self.n = len(pool_norm)
        self.starts, self.counts = starts[keep], counts[keep]
        head = arr[self.starts]
        self.ukeys = head >> np.uint64(KEY_SHIFT)
        self.utype = ((head >> np.uint64(REC_BITS)) & np.uint64(3)).astype(np.int8)
        self.weight = np.log(self.n / self.counts).astype(np.float32)
        self.recs = np.empty(n, np.int32)
        for a in range(0, n, step):
            self.recs[a:a + step] = (arr[a:a + step] & np.uint64(REC_MASK)).astype(np.int32)
        del arr

    def query(self, s1_norm, top_k=TOP_K):
        """Return DataFrame(s1_idx, pool_idx, blk_score, blk_name, blk_addr, blk_mixed, blk_nkeys, blk_rank)."""
        h, _, qidx = unpack(build_keys(s1_norm))
        pos = np.searchsorted(self.ukeys, h)
        pos[pos >= len(self.ukeys)] = 0
        hit = self.ukeys[pos] == h
        pos, qidx = pos[hit], qidx[hit]
        cnt = self.counts[pos]
        rep_q = np.repeat(qidx, cnt)
        rep_k = np.repeat(pos, cnt)
        # offsets inside each posting list
        first = np.repeat(np.cumsum(cnt) - cnt, cnt)
        within = np.arange(len(rep_q)) - first
        rep_p = self.recs[self.starts[rep_k] + within]
        code = rep_q.astype(np.int64) << 25 | rep_p.astype(np.int64)
        w = self.weight[rep_k]
        t = self.utype[rep_k]
        ucode, inv = np.unique(code, return_inverse=True)
        out = pd.DataFrame({"s1_idx": (ucode >> 25).astype(np.int32),
                            "pool_idx": (ucode & ((1 << 25) - 1)).astype(np.int32)})
        out["blk_score"] = np.bincount(inv, w).astype(np.float32)
        for ti, name in enumerate(KEY_TYPES):
            out["blk_" + name] = np.bincount(inv, np.where(t == ti, w, 0), len(ucode)).astype(np.float32)
        out["blk_nkeys"] = np.bincount(inv).astype(np.int16)
        out = out.sort_values(["s1_idx", "blk_score"], ascending=[True, False], kind="stable")
        out["blk_rank"] = out.groupby("s1_idx").cumcount().astype(np.int16)
        return out[out.blk_rank < top_k].reset_index(drop=True)
