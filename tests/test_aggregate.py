import pandas as pd

from src.eval.aggregate import summarize, verdict


def ap(tool, random, meta=(0.5, 0.5, 0.5)):
    return pd.DataFrame({"tool": tool, "random": random, "meta": meta}, index=[0, 1, 2])


def test_verdict_better():
    _, v, _, wins, n = verdict(ap([0.60, 0.61, 0.62], [0.50, 0.51, 0.50]))
    assert v == "Tool tốt hơn random" and (wins, n) == (3, 3)


def test_verdict_trend():
    _, v, *_ = verdict(ap([0.60, 0.49, 0.62], [0.50, 0.51, 0.50]))
    assert v == "Có xu hướng tool tốt hơn"


def test_verdict_no_evidence():
    _, v, *_ = verdict(ap([0.50, 0.49, 0.48], [0.50, 0.51, 0.50]))
    assert v == "Chưa có bằng chứng tool tốt hơn random"


def test_summarize_uses_latest_eval_and_sample_std():
    rows = []
    for arm in ["random", "meta", "tool"]:
        for s, v in zip([0, 1, 2], [0.1, 0.2, 0.3]):
            rows.append(dict(run_id=f"{arm}_s{s}", metric="AP50_pedestrian", value=v, evaluated_at="2026-10-20"))
    rows.append(dict(run_id="tool_s0", metric="AP50_pedestrian", value=0.9, evaluated_at="2026-10-21"))
    rows.append(dict(run_id="base", metric="AP50_pedestrian", value=0.05, evaluated_at="2026-10-19"))
    scores, arms, base = summarize(pd.DataFrame(rows))
    tool = scores[(scores.arm == "tool")].iloc[0]
    assert tool.seed0 == 0.9 and tool.n_seeds == 3
    rnd = scores[(scores.arm == "random")].iloc[0]
    assert abs(rnd["std"] - 0.1) < 1e-9                       # ddof = 1
    assert base["AP50_pedestrian"] == 0.05
