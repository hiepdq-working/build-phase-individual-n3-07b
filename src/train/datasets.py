"""Sinh danh sách ảnh và data.yaml cho từng run.

Run id: pilot, base, random_s0 … tool_s2. Run arm train trên base ∪ arm (FR-TRN-04).
"""
import yaml

from src.common.config import load_decisions
from src.common.paths import DATA_ROOT, img_path
from src.data.splits_io import read_split

LISTS = DATA_ROOT / "lists"
ARMS = ("random", "meta", "tool")


def class_names():
    return {v: k for k, v in load_decisions()["class_map"].items()}


def write_list(name, ids):
    LISTS.mkdir(parents=True, exist_ok=True)
    p = LISTS / f"{name}.txt"
    p.write_text("\n".join(str(img_path(i)) for i in ids) + "\n")
    return p


def data_yaml(name, train_ids, val_split="dev"):
    tr = write_list(f"train_{name}", train_ids)
    va = write_list(val_split, read_split(val_split))
    y = LISTS / f"data_{name}.yaml"
    y.write_text(yaml.safe_dump({"train": str(tr), "val": str(va), "names": class_names()}))
    return y


def parse_run_id(run_id):
    """-> (arm, seed). 'pilot' và 'base' là arm riêng, seed 0."""
    if run_id in ("pilot", "base"):
        return run_id, 0
    arm, _, seed = run_id.rpartition("_s")
    if arm not in ARMS or not seed.isdigit():
        raise ValueError(f"run_id không hợp lệ: {run_id} (cần pilot, base hoặc <random|meta|tool>_s<seed>)")
    return arm, int(seed)


def train_ids_for(run_id):
    arm, _ = parse_run_id(run_id)
    base = set(read_split("base"))
    if arm in ("pilot", "base"):
        return sorted(base)
    return sorted(base | set(read_split(f"arm_{arm}")))


def seed_for(run_id):
    return parse_run_id(run_id)[1]
