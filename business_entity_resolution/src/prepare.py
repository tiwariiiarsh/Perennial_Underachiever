"""Load + normalise a split once and cache it as parquet under WORK_DIR."""
import pandas as pd

from config import WORK_DIR, N_WORKERS
from io_utils import load_split
from normalize import normalize_df


def get_norm(split):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    p1, pp = WORK_DIR / f"{split}_s1_norm.parquet", WORK_DIR / f"{split}_pool_norm.parquet"
    if not (p1.exists() and pp.exists()):
        s1, pool = load_split(split)
        normalize_df(s1, N_WORKERS).to_parquet(p1)
        del s1
        normalize_df(pool, N_WORKERS).to_parquet(pp)
        del pool
    return pd.read_parquet(p1), pd.read_parquet(pp)


if __name__ == "__main__":
    import sys
    for sp in sys.argv[1:] or ["train", "test"]:
        a, b = get_norm(sp)
        print(sp, a.shape, b.shape)
