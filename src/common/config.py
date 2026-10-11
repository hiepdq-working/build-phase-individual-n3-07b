import yaml

from src.common.paths import CONFIGS


def load_yaml(name):
    with open(CONFIGS / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_decisions():
    """Bảng quyết định khóa trước (tag lock-v1). Mọi con số của thí nghiệm đọc từ đây (NFR-14, NFR-37)."""
    return load_yaml("decisions.yaml")


def load_queries():
    return load_yaml("queries.yaml")
