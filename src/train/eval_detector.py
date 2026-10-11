"""G7, G11 · Chấm last.pt trên rare slice và val2k, cùng tham số cho mọi run (FR-EVL-10 → 12).

    python -m src.train.eval_detector --run-id base

Ghi thêm dòng vào RUNS_ROOT/detector_eval.csv (Drive); tải về results/detector/detector_eval.csv.
"""
import argparse
import time

import pandas as pd

from src.common.config import load_decisions
from src.common.paths import RUNS_ROOT
from src.data.splits_io import read_split
from src.train.datasets import data_yaml

PED = 0   # class id của pedestrian trong class_map


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    cfg = load_decisions()
    e, t = cfg["eval"], cfg["train"]
    assert cfg["class_map"]["pedestrian"] == PED
    assert (RUNS_ROOT / a.run_id / "DONE").exists(), "run chưa train xong"
    w = RUNS_ROOT / a.run_id / "weights" / "last.pt"          # OI-02: luôn dùng last.pt
    model = YOLO(str(w))
    rows = []
    for split in [e["primary"]["set"], e["secondary"]["set"]]:
        y = data_yaml(f"eval_{split}", read_split("base")[:1], val_split=split)
        m = model.val(data=str(y), split="val", imgsz=t["imgsz"], batch=32, conf=e["conf"], iou=e["iou"],
                      plots=False, project=str(RUNS_ROOT / "_eval_tmp"), name=f"{a.run_id}_{split}",
                      exist_ok=True)
        if split == e["primary"]["set"]:
            ci = [int(c) for c in m.ap_class_index]
            rows.append(dict(metric="AP50_pedestrian", value=float(m.box.ap50[ci.index(PED)])))
        else:
            rows.append(dict(metric="mAP50", value=float(m.box.map50)))
        rows[-1].update(run_id=a.run_id, set=split, checkpoint="last.pt", conf=e["conf"], iou=e["iou"],
                        imgsz=t["imgsz"], evaluated_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    out = RUNS_ROOT / "detector_eval.csv"
    cols = ["run_id", "metric", "value", "set", "checkpoint", "conf", "iou", "imgsz", "evaluated_at"]
    pd.DataFrame(rows)[cols].to_csv(out, mode="a", header=not out.exists(), index=False)
    print(pd.DataFrame(rows)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
