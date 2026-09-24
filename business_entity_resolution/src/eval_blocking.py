"""Measure blocking quality on a random sample of training Source-1 records,
searched against the FULL training pool (same setting as test).

    python src/eval_blocking.py --sample 50000 --k 20 40 --show-misses 15
"""
import argparse
import time

import numpy as np

from blocking import PoolIndex
from io_utils import load_ground_truth
from prepare import get_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=50_000)
    ap.add_argument("--k", type=int, nargs="+", default=[20, 40])
    ap.add_argument("--show-misses", type=int, default=10)
    a = ap.parse_args()

    t = time.time()
    s1, pool = get_norm("train")
    gt = load_ground_truth()
    print(f"loaded {len(s1):,} S1 / {len(pool):,} pool records in {time.time() - t:.0f}s", flush=True)
    t = time.time()
    index = PoolIndex(pool)
    print(f"built index: {len(index.ukeys):,} keys in {time.time() - t:.0f}s", flush=True)

    samp = s1.sample(a.sample, random_state=1).reset_index(drop=True)
    sid, pid = samp.entity_id.to_numpy(), pool.entity_id.to_numpy()
    n_true = sum(len(gt[s]) for s in sid)
    print(f"\n{'K':>4} {'cand/S1':>8} {'pair recall':>12} {'US':>7} {'India':>7} {'query s':>8}")
    for k in a.k:
        t = time.time()
        c = index.query(samp, k)
        cand = set(zip(sid[c.s1_idx], pid[c.pool_idx]))
        hit = {"all": [0, 0]}
        misses = []
        for s, cty in zip(sid, samp.country):
            for m in gt[s]:
                ok = (s, m) in cand
                for key in ("all", cty):
                    h = hit.setdefault(key, [0, 0])
                    h[0] += ok
                    h[1] += 1
                if not ok:
                    misses.append((s, m))
        r = {kk: v[0] / max(v[1], 1) for kk, v in hit.items()}
        print(f"{k:>4} {len(c) / len(samp):>8.1f} {r['all']:>12.4f} {r.get('us', np.nan):>7.4f} "
              f"{r.get('india', np.nan):>7.4f} {time.time() - t:>8.1f}", flush=True)
    print(f"(true pairs in sample: {n_true:,})")

    if a.show_misses:
        print("\nExample missed true pairs  (S1 name | nums | addr   ||   pool name | nums | addr):")
        p, q = pool.set_index("entity_id"), samp.set_index("entity_id")
        for s, m in misses[:a.show_misses]:
            x, y = q.loc[s], p.loc[m]
            print(f"  {x['name']} | {x.nums} | {x.addr}\n    || {y['name']} | {y.nums} | {y.addr}")


if __name__ == "__main__":
    main()
