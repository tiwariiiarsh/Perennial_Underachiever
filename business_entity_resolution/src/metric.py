"""Exact challenge metric: per-Source-1 F0.5, macro-averaged (singletons included)."""


def f05(pred, true, beta=0.5):
    pred, true = set(pred), set(true)
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def macro_f05(pred_map, true_map, ids=None):
    ids = list(true_map) if ids is None else ids
    return sum(f05(pred_map.get(i, ()), true_map[i]) for i in ids) / max(len(ids), 1)


if __name__ == "__main__":
    s = f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    print(round(s, 3))  # PS example -> 0.714
    assert abs(s - 0.714) < 1e-3
