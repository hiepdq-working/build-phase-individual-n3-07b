import pandas as pd

from src.eval.verify_timing import summarize


def test_summarize_counts_unique_verified_and_wrong():
    claims = pd.DataFrame({
        "session_id": ["s1"] * 4 + ["s2"],
        "participant": ["p1"] * 4 + ["p2"],
        "mode": ["tool"] * 4 + ["browse"],
        "image_id": ["a", "a", "b", "z", "a"],
        "t_seconds": [10, 20, 30, 1000, 5],
    })
    sessions = pd.DataFrame({"session_id": ["s1", "s1"], "event": ["start", "end"],
                             "timestamp": ["2026-10-20T10:00:00", "2026-10-20T10:15:00"]})
    s = summarize(claims, sessions, rare={"a"}, limit_s=900).set_index("session_id")
    assert s.loc["s1", "n_claims"] == 3                       # claim sau 15 phút bị bỏ
    assert s.loc["s1", "n_verified_unique"] == 1
    assert s.loc["s1", "n_wrong"] == 1
    assert s.loc["s1", "duration_s"] == 900
    assert s.loc["s2", "n_verified_unique"] == 1
