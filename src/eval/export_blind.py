"""G6 · Xuất bộ chấm mù cho 5 truy vấn eval_human (FR-EVL-05, 06; NFR-09).

    python -m src.eval.export_blind vitb32_whole vitb32_whole_tiles

Gộp top-50 của mọi chỉ mục, bỏ trùng, xáo theo seed. blind_set.csv (đưa người chấm) không có
tên chỉ mục hay thứ hạng; blind_key.csv (để giải mù) giữ riêng, chỉ mở sau khi chấm xong.
"""
import sys
import time

import numpy as np
import pandas as pd

from src.common.config import load_decisions, load_queries
from src.common.paths import RESULTS
from src.data.splits_io import read_split
from src.search.engine import SearchIndex, log_run


def main(index_ids):
    qs = load_queries()["eval_human"]
    pool = set(read_split("pool"))
    key = []
    for iid in index_ids:
        ix = SearchIndex(iid)
        for q in qs:
            t0 = time.time()
            hits = ix.search(ix.encode_text([q["text"]]), n=50, allowed=pool)
            log_run(ix, "eval_human", q["id"], [q["text"]], hits, time.time() - t0)
            key += [dict(query_id=q["id"], image_id=h, index_id=iid) for h in hits.image_id]
    key = pd.DataFrame(key)
    pairs = key[["query_id", "image_id"]].drop_duplicates()
    pairs = pairs.sample(frac=1, random_state=load_decisions()["seeds"]["judging_shuffle"]).reset_index(drop=True)
    pairs.insert(0, "pair_id", np.arange(1, len(pairs) + 1))
    qmap = {q["id"]: q for q in qs}
    out = RESULTS / "relevance"
    out.mkdir(parents=True, exist_ok=True)
    pairs.assign(query_text=pairs.query_id.map(lambda i: qmap[i]["text"]),
                 definition=pairs.query_id.map(lambda i: qmap[i].get("definition", "")))[
        ["pair_id", "query_text", "definition", "image_id"]].to_csv(out / "blind_set.csv", index=False)
    key.merge(pairs, on=["query_id", "image_id"]).to_csv(out / "blind_key.csv", index=False)
    print(f"{len(pairs)} cặp cần chấm -> {out / 'blind_set.csv'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Cách dùng: python -m src.eval.export_blind <index_id> [<index_id> ...]")
    main(sys.argv[1:])
