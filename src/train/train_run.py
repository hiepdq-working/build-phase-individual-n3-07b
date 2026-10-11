"""G7, G10 · Train một run YOLOv8s (FR-TRN-01 → 10). Chạy trên Colab qua notebooks/02_train.ipynb.

    python -m src.train.train_run --run-id pilot --epochs 3 --fraction 0.2   # pilot đo thời gian + thử resume
    python -m src.train.train_run --run-id base
    python -m src.train.train_run --run-id tool_s1

Checkpoint lưu lên RUNS_ROOT/<run_id>/weights/last.pt sau mỗi epoch; chạy lại cùng run_id thì tự resume.
"""
import argparse
import hashlib
import json
import re
import subprocess
import time

from src.common.config import load_decisions
from src.common.paths import CONFIGS, REPO, RUNS_ROOT
from src.train.datasets import data_yaml, seed_for, train_ids_for


def pinned_version(package, req_file=REPO / "requirements-colab.txt"):
    m = re.search(rf"^{re.escape(package)}==([\w.]+)", req_file.read_text(), re.M)
    return m.group(1) if m else None


def check_pins(strict):
    """FR-TRN-06: dừng nếu ultralytics khác bản đã pin trong requirements-colab.txt."""
    import ultralytics
    pin = pinned_version("ultralytics")
    if pin is None:
        msg = "requirements-colab.txt chưa pin ultralytics==<version> (Bước 4.2)"
        if strict:
            raise SystemExit(msg)
        print("! " + msg)
    elif ultralytics.__version__ != pin:
        raise SystemExit(f"ultralytics {ultralytics.__version__} khác bản pin {pin}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--epochs", type=int, help="chỉ dùng cho pilot")
    ap.add_argument("--fraction", type=float, default=1.0, help="chỉ dùng cho pilot")
    a = ap.parse_args()
    rid = a.run_id
    pilot = rid == "pilot"
    check_pins(strict=not pilot)

    import torch
    import ultralytics
    from ultralytics import YOLO

    cfg = load_decisions()
    t = cfg["train"]
    if not pilot:
        assert a.epochs is None and a.fraction == 1.0, "run thật không được ghi đè tham số"
        assert cfg["locked_on"] and t["epochs"], "decisions.yaml chưa khóa (lock-v1) hoặc chưa có epochs"
    epochs = a.epochs or t["epochs"]
    ids = train_ids_for(rid)
    expect = cfg["sizes"]["base"] + (0 if rid in ("pilot", "base") else cfg["sizes"]["arm_k"])
    assert len(ids) == len(set(ids)) == expect, f"{rid}: {len(ids)} ảnh, cần {expect}"
    assert all(i.startswith("train/") for i in ids), "có ảnh val trong tập train"
    y = data_yaml(rid, ids)

    d = RUNS_ROOT / rid
    d.mkdir(parents=True, exist_ok=True)
    if (d / "DONE").exists():
        print(f"{rid} đã xong từ trước")
        return
    env = dict(run_id=rid, time=time.strftime("%Y-%m-%dT%H:%M:%S"), torch=torch.__version__,
               ultralytics=ultralytics.__version__,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               git_commit=subprocess.getoutput(f"git -C {REPO} rev-parse HEAD"),
               decisions_sha256=hashlib.sha256((CONFIGS / "decisions.yaml").read_bytes()).hexdigest(),
               n_train=len(ids), seed=seed_for(rid), epochs=epochs)
    with open(d / "env.jsonl", "a") as f:                  # mỗi phiên Colab một dòng (FR-TRN-10)
        f.write(json.dumps(env) + "\n")

    last = d / "weights" / "last.pt"
    if last.exists():
        print("Resume từ", last)                           # FR-TRN-09
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(t["model"]).train(
            data=str(y), epochs=epochs, imgsz=t["imgsz"], batch=t["batch"], workers=t["workers"],
            optimizer=t["optimizer"], lr0=t["lr0"], seed=seed_for(rid), deterministic=True,
            val=t["val_during_train"], fraction=a.fraction, cache="disk",
            project=str(RUNS_ROOT), name=rid, exist_ok=True, plots=False)
    (d / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    print(f"{rid} xong")


if __name__ == "__main__":
    main()
