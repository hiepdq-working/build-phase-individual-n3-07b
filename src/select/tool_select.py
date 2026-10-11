"""G9 · Chọn 500 ảnh cho arm_tool (FR-SEL-03 → 05, FR-SRC-09, 10). Chạy trên Mac.

    python -m src.select.tool_select

Lấy 1.500 ảnh pool có điểm CLIP cao nhất với các prompt tool (model chính, configs/main_model.yaml),
rồi giữ 500 ảnh uncertainty cao nhất; hòa thì xét điểm CLIP rồi image_id (OI-11).
Script KHÔNG đọc metadata hay nhãn GT (FR-SEL-05). Chạy lần hai phải cho cùng sha256.
"""
import time

import pandas as pd

from src.common.config import load_decisions, load_queries, load_yaml
from src.common.paths import CONFIGS, RESULTS
from src.data.splits_io import read_split, write_split
from src.search.engine import SearchIndex, log_run


def select_top(cand, k):
    """cand: image_id, score, uncertainty -> cand có cột selected (k ảnh uncertainty cao nhất)."""
    cand = cand.sort_values(["uncertainty", "score", "image_id"], ascending=[False, False, True]).copy()
    cand["selected"] = False
    cand.iloc[:k, cand.columns.get_loc("selected")] = True
    return cand


def require_locked(cfg):
    """FR-DAT-11: tag lock-v1 phải có trước SEARCH_RUN tool_select đầu tiên."""
    if not cfg.get("locked_on"):
        raise SystemExit("decisions.yaml chưa khóa (locked_on trống) — làm Bước 4.4 (điền locked_on, "
                         "gắn tag lock-v1) trước khi chạy tool_select (FR-DAT-11)")


def main():
    cfg = load_decisions()
    require_locked(cfg)
    if not (CONFIGS / "main_model.yaml").exists():
        raise SystemExit("Chưa có configs/main_model.yaml — làm Bước 5.3 (chọn model chính) và commit trước")
    main_cfg = load_yaml("main_model.yaml")
    prompts = load_queries()["tool"]
    if not 3 <= len(prompts) <= 5:
        raise SystemExit("tool cần 3–5 prompt")                     # FR-SRC-10
    ix = SearchIndex(main_cfg["tool_select_index"])
    assert ix.meta["model"] == main_cfg["main_model"], "chỉ mục không khớp model chính"
    n_cand = cfg["sizes"]["tool_candidates"]
    t0 = time.time()
    hits = ix.search(ix.encode_text(prompts), n=n_cand, allowed=set(read_split("pool")))
    rid = log_run(ix, "tool_select", "TOOL", prompts, hits, time.time() - t0)
    assert hits.image_id.nunique() == n_cand

    unc = pd.read_csv(RESULTS / "uncertainty" / "base_pool.csv")[["image_id", "uncertainty"]]
    cand = hits.merge(unc, on="image_id", how="left")
    assert cand.uncertainty.notna().all(), "có ảnh thiếu uncertainty"
    cand = select_top(cand, cfg["sizes"]["arm_k"])
    out = RESULTS / "select"
    out.mkdir(exist_ok=True)
    cand.assign(search_run_id=rid).to_csv(out / "tool_candidates.csv", index=False)
    write_split("arm_tool", cand[cand.selected].image_id, note=f"search_run_id={rid}")


if __name__ == "__main__":
    main()
