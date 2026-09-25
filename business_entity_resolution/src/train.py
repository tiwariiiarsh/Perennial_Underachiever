"""Train the pair classifier and tune the decision rule.

* A random sample of Source-1 training entities is blocked against the FULL
  training pool (same setting as test).
* LightGBM with GroupKFold by Source-1 id gives out-of-fold probabilities;
  isotonic regression calibrates them; decision parameters are grid-searched
  on the out-of-fold macro F0.5 (an honest validation estimate).
* A final model is refit on all sampled pairs and saved with the calibrator,
  decision parameters and a metrics report.
"""
import itertools
import json
import pickle
import time

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from blocking import PoolIndex
from config import LGB_PARAMS, MODEL_DIR, N_FOLDS, SEED, TRAIN_SAMPLE_S1, TOP_K
from decide import decide, select, DEFAULT_PARAMS
from features import pair_features
from io_utils import load_ground_truth
from metric import macro_f05
from prepare import get_norm


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def per_country(pred, gt, ids, country):
    out = {}
    for c in sorted(set(country)):
        sel = [i for i, cc in zip(ids, country) if cc == c]
        out[c] = round(macro_f05(pred, gt, sel), 4)
    return out


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    s1, pool = get_norm("train")
    gt = load_ground_truth()
    log("building pool index")
    index = PoolIndex(pool)
    samp = s1.sample(min(TRAIN_SAMPLE_S1, len(s1)), random_state=SEED).reset_index(drop=True)
    del s1
    log("blocking", len(samp))
    cands = index.query(samp, TOP_K)
    del index
    pid = pool.entity_id.to_numpy()
    sid = samp.entity_id.to_numpy()
    y = np.fromiter((p in gt[s] for s, p in zip(sid[cands.s1_idx], pid[cands.pool_idx])),
                    bool, len(cands)).astype(np.int8)
    n_true = sum(len(gt[s]) for s in sid)
    blk_recall = y.sum() / n_true
    log(f"candidates={len(cands)} per_s1={len(cands)/len(samp):.1f} pair_recall={blk_recall:.4f}")
    log("features")
    X = pair_features(cands, samp, pool)
    feats = list(X.columns)

    oof = np.zeros(len(X))
    for fold, (tr, va) in enumerate(GroupKFold(N_FOLDS).split(X, y, cands.s1_idx)):
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        log(f"fold {fold} auc={roc_auc_score(y[va], oof[va]):.5f}")
    auc, ap = roc_auc_score(y, oof), average_precision_score(y, oof)
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    cands = cands[["s1_idx", "pool_idx"]].copy()
    cands["prob"] = iso.predict(oof)

    def score(params):
        sel = decide(cands, params)
        pred = {}
        for s, p in zip(sid[sel.s1_idx], pid[sel.pool_idx]):
            pred.setdefault(s, []).append(p)
        return macro_f05(pred, gt, list(sid)), pred

    ablation = {}
    naive = cands[cands.prob >= 0.5]
    pred05 = {}
    for s, p in zip(sid[naive.s1_idx], pid[naive.pool_idx]):
        pred05.setdefault(s, []).append(p)
    ablation["threshold_0.5_only"] = macro_f05(pred05, gt, list(sid))
    ablation["predict_nothing"] = macro_f05({}, gt, list(sid))
    best, best_p = -1, None
    for margin, mp, miss in itertools.product([0.0, 0.05, 0.1, 0.15], [0.6, 0.65, 0.7, 0.75, 0.85], [0.3, 0.4, 0.5, 0.6]):
        prm = dict(margin=margin, min_prob=mp, miss=miss)
        f = score(prm)[0]
        if f > best:
            best, best_p = f, prm
    f_best, pred = score(best_p)
    sel = select(cands, best_p["min_prob"], best_p["miss"])
    pno = {}
    for s_, p_ in zip(sid[sel.s1_idx], pid[sel.pool_idx]):
        pno.setdefault(s_, []).append(p_)
    ablation["expectedF_without_one_owner"] = macro_f05(pno, gt, list(sid))
    ablation["full_decision"] = f_best
    ablation = {k: round(v, 4) for k, v in ablation.items() if v is not None}
    log("best params", best_p, "oof macro F0.5", round(f_best, 4))

    log("final fit")
    final = lgb.LGBMClassifier(**LGB_PARAMS).fit(X, y)
    final.booster_.save_model(str(MODEL_DIR / "lgb.txt"))
    pickle.dump(iso, open(MODEL_DIR / "iso.pkl", "wb"))
    imp = dict(sorted(zip(feats, final.booster_.feature_importance("gain").round(1).tolist()),
                      key=lambda kv: -kv[1]))
    metrics = dict(
        n_train_s1=len(samp), top_k=TOP_K, candidates_per_s1=round(len(cands) / len(samp), 2),
        blocking_pair_recall=round(float(blk_recall), 4), pair_auc=round(auc, 5), pair_ap=round(ap, 5),
        oof_macro_f05=round(f_best, 4), per_country_f05=per_country(pred, gt, list(sid), samp.country),
        decision_params=best_p, ablation=ablation, feature_importance=imp, features=feats,
    )
    json.dump(metrics, open(MODEL_DIR / "metrics.json", "w"), indent=2)
    json.dump({**DEFAULT_PARAMS, **best_p}, open(MODEL_DIR / "decision.json", "w"))
    log(json.dumps({k: v for k, v in metrics.items() if k not in ("feature_importance", "features")}))


if __name__ == "__main__":
    main()
