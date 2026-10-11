"""Mọi đường dẫn đọc từ biến môi trường, không viết cứng (NFR-34).

Cùng một code chạy trên Mac (mặc định: data/, index/, runs/ trong repo) và trên Colab
(đặt DATA_ROOT, INDEX_ROOT, RUNS_ROOT trỏ tới ổ Colab / Google Drive).
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO / "data"))      # chứa bdd100k/ và meta/
INDEX_ROOT = Path(os.environ.get("INDEX_ROOT", REPO / "index"))    # chứa các chỉ mục CLIP
RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", REPO / "runs"))       # checkpoint + log huấn luyện

BDD = DATA_ROOT / "bdd100k"
IMG_DIR = {"train": BDD / "images/100k/train", "val": BDD / "images/100k/val"}
LABEL_JSON = {"train": BDD / "labels/det_20/det_train.json", "val": BDD / "labels/det_20/det_val.json"}
META = DATA_ROOT / "meta"
YOLO_LABELS = BDD / "labels/100k"   # Ultralytics tìm nhãn bằng cách đổi /images/ thành /labels/

SPLITS = REPO / "splits"
CONFIGS = REPO / "configs"
RESULTS = REPO / "results"
REPORT = REPO / "report"


def img_path(image_id: str) -> Path:
    """image_id có dạng 'train/xxxx.jpg' hoặc 'val/xxxx.jpg'."""
    split, name = image_id.split("/", 1)
    return IMG_DIR[split] / name
