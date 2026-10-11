#!/usr/bin/env bash
# Bước 9.1 — tái tạo 3 con số từ file đã commit (FR-EVL-19, NFR-13).
# Cần data/meta/*.parquet và index/ (không nằm trong git): trên bản clone sạch thì symlink vào.
set -euo pipefail
python -m src.data.check_splits
echo "== Con số 1a: P@50 tự động =="
python -m src.eval.eval_search vitb32_whole vitb32_whole_tiles --no-log
echo "== Con số 1b: P@50 chấm mù =="
python -m src.eval.score_blind
echo "== Con số 2: timing =="
python -m src.eval.verify_timing
echo "== Con số 3: AP50 9 run =="
python -m src.eval.aggregate
echo
echo "Đối chiếu: git status phải chỉ báo thay đổi ở thời gian tạo hình (nếu có); số khớp báo cáo đến 3 chữ số."
