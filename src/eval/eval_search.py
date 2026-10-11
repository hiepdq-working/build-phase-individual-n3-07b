"""G6 · Precision@50 tự động trên 20 truy vấn eval_auto (FR-EVL-02, 03) — con số 1.

    python -m src.eval.eval_search vitb32_whole vitb32_whole_tiles
    python -m src.eval.eval_search vitb32_whole vitb32_whole_tiles --no-log   # dùng trong reproduce.sh

Commit configs/queries.yaml + src/eval/gt_rules.py TRƯỚC lần chạy đầu tiên.
"""
import argparse
import time

import pandas as pd

from src.common.config import load_queries
from src.common.paths import RESULTS
from src.data.splits_io import read_split
from src.eval.gt_rules import load_table, positives
from src.search.engine import SearchIndex, log_run

N = 50


def main(index_ids, log=True):
    qs = load_queries()["eval_auto"]
    pool = set(read_split("pool"))
    table = load_table()
    pos = {q["id"]: positives(q["rule"], table, pool) for q in qs}
    out = RESULTS / "search"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"query_id": k, "n_positive_in_pool": len(v), "under_50": len(v) < N}
                  for k, v in pos.items()]).to_csv(out / "positives_per_query.csv", index=False)
    rows = []
    for iid in index_ids:
        ix = SearchIndex(iid)
        for q in qs:
            t0 = time.time()
            hits = ix.search(ix.encode_text([q["text"]]), n=N)
            if log:
                log_run(ix, "eval", q["id"], [q["text"]], hits, time.time() - t0)
            rows.append(dict(index_id=iid, query_id=q["id"],
                             p_at_50=len(set(hits.image_id) & pos[q["id"]]) / N))
    df = pd.DataFrame(rows)
    df.to_csv(out / "p_at_50_auto.csv", index=False)
    table = df.pivot(index="query_id", columns="index_id", values="p_at_50")
    table.loc["MEAN"] = table.mean()
    table.to_csv(out / "p_at_50_auto_table.csv")
    print(table.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("index_ids", nargs="+")
    ap.add_argument("--no-log", action="store_true", help="không ghi runs.csv / hits (khi tái tạo số)")
    a = ap.parse_args()
    main(a.index_ids, log=not a.no_log)
