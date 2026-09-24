"""TSV reading/writing (always tab-separated, all columns as strings)."""
import pandas as pd

from config import DATA_DIR


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def load_split(split):
    """Return (source1, pool) where pool = Source 2 + Source 3 concatenated."""
    d = DATA_DIR / split
    s1 = read_tsv(d / f"{split}_source1.tsv")
    pool = pd.concat([read_tsv(d / f"{split}_source2.tsv"), read_tsv(d / f"{split}_source3.tsv")],
                     ignore_index=True)
    return s1, pool


def load_ground_truth():
    gt = read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv")
    return dict(zip(gt.source1_entity_id,
                    (set(x.split(",")) if x else set() for x in gt.matched_entity_ids)))


def write_id_lists(path, s1_ids, lists, col):
    """Write one row per Source-1 id; `lists` maps s1 id -> iterable of ids."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s in s1_ids:
            ids = list(dict.fromkeys(lists.get(s, ())))
            f.write(f"{s}\t{','.join(ids)}\n")
