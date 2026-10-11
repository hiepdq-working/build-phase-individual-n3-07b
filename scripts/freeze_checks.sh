#!/usr/bin/env bash
# Bước 9.2 — cổng kiểm tra trước freeze (AC-06, AC-07). Ghi toàn bộ output vào results/freeze_checks.txt
set -uo pipefail
out=results/freeze_checks.txt
fail=0
{
  echo "# Freeze checks — $(date '+%Y-%m-%dT%H:%M:%S') — commit $(git rev-parse --short HEAD)"
  echo; echo "## check_splits"
  python -m src.data.check_splits || fail=1
  echo; echo "## NFR-01: git diff lock-v1 -- configs/decisions.yaml (phải rỗng)"
  if git rev-parse -q --verify refs/tags/lock-v1 >/dev/null; then
    d=$(git diff lock-v1 -- configs/decisions.yaml); [ -z "$d" ] && echo "rỗng — OK" || { echo "$d"; fail=1; }
  else echo "CHƯA có tag lock-v1"; fail=1; fi
  echo; echo "## NFR-36: pytest"
  python -m pytest -q || fail=1
  echo; echo "## NFR-39: file dữ liệu trong git (phải rỗng)"
  f=$(git ls-files | grep -E '\.(jpg|jpeg|npy|faiss|pt|parquet|zip)$' | grep -v '^report/figures/' || true)
  [ -z "$f" ] && echo "rỗng — OK" || { echo "$f"; fail=1; }
  echo; echo "## NFR-42: email trong results/ (phải rỗng)"
  e=$(grep -rnE "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}" results/ | grep -v freeze_checks || true)
  [ -z "$e" ] && echo "rỗng — OK" || { echo "$e"; fail=1; }
  echo; echo "## NFR-03: tên rare slice / val2k trong mã G4–G9 (phải rỗng)"
  g=$(grep -rnE "rare_val|val2k" src/search src/uncertainty src/select || true)
  [ -z "$g" ] && echo "rỗng — OK" || { echo "$g"; fail=1; }
  echo; echo "## Thứ tự commit (đối chiếu với results/runs_registry.csv và results/search/runs.csv)"
  git log --format="%h %ci %s" -- configs/decisions.yaml configs/queries.yaml src/eval/gt_rules.py \
      configs/main_model.yaml splits/arm_tool.txt splits/arm_random.txt splits/arm_meta.txt | head -20
  git tag -l --format="%(refname:short) %(creatordate:iso)" lock-v1
  echo; [ $fail -eq 0 ] && echo "KẾT QUẢ: PASS" || echo "KẾT QUẢ: FAIL"
} 2>&1 | tee "$out"
exit $fail
