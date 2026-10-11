# Truy vết SRD → repo

Yêu cầu **Must** của SRD v0.3 và nơi đáp ứng trong repo. Các mục *Could* đã bỏ khi làm một mình được ghi ở cuối.

## Dữ liệu (FR-DAT)

| ID | Nơi đáp ứng | Cách kiểm |
|---|---|---|
| DAT-01, 02 | `src/data/build_meta.py` | assert image_id duy nhất; output in số lớp |
| DAT-03 | `src/data/bdd_to_yolo.py` | `tests/test_yolo_labels.py` (round-trip 50 ảnh) |
| DAT-04 → 08 | `src/data/make_splits.py`, `splits_io.py` | chạy 2 lần, sha256 trong `splits/registry.csv` không đổi |
| DAT-09 | `src/data/check_splits.py` | `tests/test_check_splits.py` |
| DAT-10 | `scripts/pack_labels.sh`, notebook 02 ô 3 | checksum trong `docs/bdd100k.sha256` |
| DAT-11 | `configs/decisions.yaml` + tag `lock-v1` | `scripts/freeze_checks.sh`, CI |

## Chỉ mục (FR-IDX) và tìm kiếm (FR-SRC)

| ID | Nơi đáp ứng | Cách kiểm |
|---|---|---|
| IDX-01 | `src/search/tiling.py` | `tests/test_tiling.py` |
| IDX-02 | `embed.py` dừng nếu thiếu `clip.preprocess` | — |
| IDX-03 → 08 | `src/search/embed.py`, `check_index.py` | `check_index` in OK |
| IDX-09 | 2 chỉ mục ViT-B-32 bắt buộc; SigLIP nếu kịp | `ls index/` |
| SRC-01, 02 | `SearchIndex._load`, `encode_image` dùng model/preprocess của meta.json | `tests/test_search.py` (lệch chiều bị từ chối) |
| SRC-03 | `encode_text` nhận 1–5 prompt | `tests/test_search.py` |
| SRC-04 → 08 | `engine.rank_hits` | `tests/test_search.py` |
| SRC-09 | quét toàn bộ vector (`k_vectors` trong runs.csv) | `results/search/runs.csv` |
| SRC-10 | `tool_select.py` kiểm 3–5 prompt | `tests/test_configs.py` |
| SRC-11 | `engine.log_run` | `results/search/runs.csv`, `hits/` |

## Độ bất định và chọn ảnh (FR-SEL)

| ID | Nơi đáp ứng | Cách kiểm |
|---|---|---|
| SEL-01, 02 | `src/uncertainty/run_pool.py`, `score.py` | `tests/test_uncertainty.py`; assert 40.000 dòng ∈ [0, 1] |
| SEL-03 → 05 | `src/select/tool_select.py` (không đọc metadata/GT) | `tests/test_selection.py`; chạy 2 lần cùng sha256 |
| SEL-06, 07 | `src/select/make_arms.py` | `check_splits` |
| SEL-08 | commit 3 arm trước run arm đầu tiên | `freeze_checks.sh` (git log) so với registry |
| SEL-09 (Should) | `src/select/arm_report.py` | `results/select/arm_composition.csv` |

## Huấn luyện (FR-TRN)

| ID | Nơi đáp ứng |
|---|---|
| TRN-01 | `notebooks/02_train.ipynb` — chỉ sửa ô `RUN_ID` |
| TRN-02 → 05 | `src/train/train_run.py`, `datasets.py` (assert 5.000 / 5.500 ảnh, 0 ảnh val, tham số từ decisions.yaml) |
| TRN-06 | `train_run.check_pins` + `requirements-colab.txt` |
| TRN-07 | notebook 02 ô 3: chép zip sang ổ Colab, in checksum |
| TRN-08, 09 | Ultralytics `project=RUNS_ROOT` (Drive) + `resume=True` |
| TRN-10 | `runs/<id>/env.jsonl` (phiên bản, GPU, git commit, hash decisions.yaml) |
| TRN-11, 12 | `results/runs_registry.csv` (thay Google Sheet) |

## Đánh giá (FR-EVL)

| ID | Nơi đáp ứng |
|---|---|
| EVL-01, 02 | `configs/queries.yaml` (eval_auto) + `src/eval/gt_rules.py`; `positives_per_query.csv` |
| EVL-03 | `src/eval/eval_search.py` → `p_at_50_auto_table.csv` |
| EVL-04 | `configs/main_model.yaml` commit trước tool_select |
| EVL-05 → 08 | `queries.yaml` (eval_human), `export_blind.py`, `app/judge.py`, `score_blind.py` |
| EVL-09 (Could) | kappa trong `score_blind.py` (nếu có j2) |
| EVL-10 → 12 | `src/train/eval_detector.py` (last.pt, cùng tham số) |
| EVL-13 → 15 | `src/eval/aggregate.py` |
| EVL-16 → 18 | `app/app.py` (Timing) + `src/eval/verify_timing.py` |
| EVL-19 | `scripts/reproduce.sh` |

## App (FR-APP)

APP-01, 02, 04 → 08, 10 → 12 và APP-13 → 16 (timing, vốn là Could nhưng dùng cho con số 2): `app/app.py`. APP-17 (chấm mù): `app/judge.py`.

## Phi chức năng chính

| NFR | Kiểm bằng |
|---|---|
| 01 | CI + `freeze_checks.sh` (`git diff lock-v1 -- configs/decisions.yaml`) |
| 02 | `check_splits` |
| 03 | `freeze_checks.sh` (grep tên rare slice/val2k trong `src/search`, `src/uncertainty`, `src/select`) |
| 05 | diff `args.yaml` giữa các run (Bước 6.1) |
| 08 | `embed.py` tắt `requires_grad`; không có optimizer cho CLIP |
| 09 | `blind_set.csv` không có cột cấu hình/thứ hạng; `judge.py` |
| 13 | `scripts/reproduce.sh` |
| 15 | `requirements-mac.lock.txt`, `requirements-colab.txt` |
| 34 | `src/common/paths.py` (biến môi trường) |
| 36 | `tests/` |
| 39 | `.gitignore` + CI |
| 42 | `freeze_checks.sh` (grep email trong results/) |

## Đã bỏ (Could) khi làm một mình

IDX-10 (model CLIP thứ ba), APP-03 (ảnh tương tự), APP-09 (khung best tile), CVT-01 → 03 (CVAT). Google Sheet run_tracker được thay bằng `results/runs_registry.csv`. OpenVINO được thay bằng PyTorch CPU/MPS trên Mac.
