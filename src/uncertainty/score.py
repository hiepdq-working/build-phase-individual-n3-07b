"""G8 · Công thức độ bất định (FR-SEL-02, OI-01).

uncertainty = số box có conf ∈ [conf_min, conf_max] / số box có conf ≥ conf_min.
Ảnh không có box nào ≥ conf_min lấy empty_value.
"""
import numpy as np


def uncertainty(conf, cmin=0.2, cmax=0.5, empty=0.0):
    conf = np.asarray(conf, dtype=float)
    conf = conf[conf >= cmin]
    if len(conf) == 0:
        return float(empty)
    return float((conf <= cmax).sum() / len(conf))
