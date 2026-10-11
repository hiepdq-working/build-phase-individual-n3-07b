# Kế hoạch làm một mình — 3 tuần (buổi tối + cuối tuần)

Nguồn: *N3-07B làm một mình — checklist từng bước* (11/10/2026), *Kế hoạch chi tiết*, *SRD v0.3*.
Tích ô khi phần **Kiểm tra** của bước đã qua. Ghi ngày thật vào cột cuối.

## Thay đổi so với SRD khi làm một mình

| Mục trong SRD | Làm một mình | Lý do |
|---|---|---|
| Laptop Windows + OpenVINO (C-03) | MacBook M2, PyTorch CPU/MPS | Máy của mình; ghi vào báo cáo |
| Google Sheet run_tracker | `results/runs_registry.csv` (Sheet là tùy chọn) | Một người chạy, không cần chia tài khoản |
| Họp hằng ngày, PR có người duyệt | Bỏ; vẫn commit thường xuyên, tag `lock-v1`, `freeze` | Không có người duyệt |
| 4 chỉ mục (2 model × 2 tiling) | Bắt buộc 2 chỉ mục ViT-B-32; SigLIP nếu kịp W2 | Tiết kiệm GPU; ghi hạn chế nếu bỏ |
| CVAT, nút "ảnh tương tự", khung best tile | Bỏ (đều là Could) | Không đủ thời gian |
| Chấm mù và đo thời gian | Vẫn làm, nhờ 1–2 bạn giúp khoảng 1 giờ | Tự chấm / tự đo sẽ thiên lệch |

## Lịch

| Tuần | Việc | Mốc phải giữ |
|---|---|---|
| Trước W1 (2–3 tối) | Bước 0: môi trường, repo, tải BDD100K | Dữ liệu nằm trên Mac và Drive |
| W1 | Bước 1 → 4: dữ liệu, chỉ mục ViT-B-32, P@50, pilot + detector gốc | **Mốc 1 (CN W1):** bảng P@50 whole vs tiles, AP50 của base, tìm bằng câu chạy được |
| W2 | Bước 5 → 6 (đưa 9 run vào chạy), Bước 7 app, SigLIP nếu kịp | **Mốc 2 (T4 W2):** run arm đầu tiên đã chạy, 9 run có lịch |
| W3 | Chấm 9 run, Bước 8, Bước 9 | **Freeze T4 W3** · video T5 · tập thuyết trình T6–T7 |

## Checklist theo bước

### Bước 0 — Môi trường và dữ liệu
- [ ] Python 3.11, VS Code (Python + Jupyter), git config
- [ ] `.venv`, `pip install -r requirements-mac.txt`, `requirements-mac.lock.txt`; torch in `True` cho MPS
- [ ] Repo trên GitHub (`build-phase-individual-n3-07b`), push
- [ ] `bash scripts/download_bdd.sh`: 70.000 / 10.000 ảnh, `docs/bdd100k.sha256`
- [ ] `docs/data_sources.md` điền nguồn, ngày tải, dung lượng, SHA-256, số ảnh
- [ ] `notebooks/00_setup_drive.ipynb`: hash Drive = hash Mac
- [ ] Tài khoản Kaggle đã xác minh số điện thoại (dự phòng GPU)

### Bước 1 — G0 (Mac, 1–2 tối)
- [ ] Rà `configs/decisions.yaml` (được sửa tới trước lock-v1)
- [ ] `make meta`: không lỗi assert, có lớp `pedestrian`; ghi số ảnh thiếu JSON
- [ ] `make yolo`; `pytest -q tests/test_yolo_labels.py` (test round-trip 50 ảnh chạy khi có dữ liệu)
- [ ] Mở 5 ảnh (có ảnh đêm), vẽ box từ file txt để nhìn bằng mắt
- [ ] `make splits` (chạy 2 lần → sha256 không đổi); chép output vào `docs/data_sources.md`
- [ ] `make check` in OK; `make pack`; kéo zip lên `MyDrive/n3-07b/raw/`; commit
- **Kiểm tra:** check_splits OK · pytest pass · rare slice ≥ 150 (hoặc ghi mức fallback) · pool ≥ 1.000 ảnh night ∧ (rainy ∨ snowy)

### Bước 2 — G1–G3 (Colab)
- [ ] `01_embed.ipynb` ô 5: đo crop/s → `docs/pilot.md`
- [ ] Ô 6: 40.000 ảnh; xoá `vitb32_test_*`
- [ ] Tải `vitb32_whole`, `vitb32_whole_tiles` về `index/`; `check_index` cả hai in OK; ghi cosine Mac vs Colab
- **Kiểm tra:** 40.000 và 280.000 vector, mỗi thư mục đủ 4 file

### Bước 3 — G4–G6 (Mac)
- [ ] Kiểm tên giá trị thật (`parking lot`, `dawn/dusk`) khớp `gt_rules.py`
- [ ] **Commit `configs/queries.yaml` + `src/eval/gt_rules.py` trước**
- [ ] `make eval-search`; xem `positives_per_query.csv` (đánh dấu câu < 50 ảnh dương); commit `results/search/`
- **Kiểm tra:** bảng 20 × 2 có dòng MEAN; chạy lại ra cùng bảng

### Bước 4 — G7 (Colab) — Mốc 1
- [ ] Pin `requirements-colab.txt` (4 dòng `==`), commit
- [ ] Pilot 3 epoch / 20% base; thử ngắt → resume; ghi thời gian epoch → `docs/pilot.md`
- [ ] Chọn `E = floor(90 / T)` (bội 5, ≥ 10); sửa `train.epochs`, `locked_on`
- [ ] `git tag lock-v1 && git push origin lock-v1`
- [ ] Run `base`; cập nhật `results/runs_registry.csv`; tải `detector_eval.csv` về `results/detector/`
- **Kiểm tra (AC-01):** AP50 base trên rare slice · bảng P@50 · tìm bằng câu chạy được · tag lock-v1 sớm hơn run base

### Bước 5 — G8–G9 (đầu W2)
- [ ] `02_train.ipynb` ô 8 (`run_pool`), tải `base_pool.csv` + `params.json` về `results/uncertainty/`
- [ ] Tạo `configs/main_model.yaml` theo `main_model_rule`, commit
- [ ] `make tool-select` (2 lần → sha256 không đổi), `make arms`, `make arm-report`
- [ ] **Commit 3 arm + registry trước run arm đầu tiên**
- **Kiểm tra:** check_splits OK với 3 arm · arm_composition.csv 3 dòng · không sửa prompt sau khi xem bảng

### Bước 6 — G10–G11 — Mốc 2
- [ ] 9 dòng `todo` có ngày dự kiến trong registry (định nghĩa "queued", OI-23)
- [ ] Chạy xen kẽ theo seed: `*_s0` → `*_s1` → `*_s2`
- [ ] Diff `args.yaml` giữa các run chỉ khác seed/data/name
- [ ] **T5 W2:** còn ≥ 5/9 run chưa chạy → mua Colab Pro (hoặc Kaggle), ghi vào cột note
- [ ] Đủ 9 run → `make aggregate`; kiểm số trên biểu đồ khớp CSV
- [ ] W3: chấm lại 1 run, lệch AP50 ≤ 0,002 → `docs/pilot.md`
- **Kiểm tra (AC-05):** 9/9 done · detector_scores.csv 3 arm × 2 metric · 3 giá trị tool − random · kết luận

### Bước 7 — App (song song W2)
- [ ] 6 tính năng với câu `a pedestrian at night in the rain`
- [ ] 5 ca lỗi (rỗng, file không phải ảnh, image_id sai, meta.json thiếu, 6 prompt)
- [ ] Tắt Wi-Fi vẫn chạy
- [ ] `docs/app_perf.md`: 20 truy vấn, khởi động, RAM
- **Kiểm tra (AC-08):** 6 tính năng offline · 5 ca lỗi pass · có số đo hiệu năng

### Bước 8 — Chấm mù và timing (cuối W2 / đầu W3)
- [ ] `make blind`; đưa bạn đọc `docs/judge_guideline.md`
- [ ] j1 chấm hết, j2 chấm ≥ 20%; ảnh chụp xác nhận "không thấy cấu hình/thứ hạng" → `docs/`
- [ ] `make score-blind`; phân xử `disagreements.csv` → `results/relevance/adjudicated.csv` nếu có
- [ ] Timing: p1 duyệt tay → tool, p2 tool → duyệt tay; `make timing`
- **Kiểm tra (AC-03, AC-04):** P@50 chấm mù · mỗi chế độ ≥ 1 phiên 15 phút · số verified + claim sai

### Bước 9 — Freeze và nộp (W3)
- [ ] `make reproduce` trên bản clone sạch (symlink `data/`, `index/`)
- [ ] `make freeze-checks` → PASS; đối chiếu thời điểm commit
- [ ] `results/runs_registry.csv` lần cuối; `docs/open_issues.md` mọi OI có trạng thái
- [ ] `git tag freeze && git push origin freeze`
- [ ] `report/report.md`, slide, video dự phòng (Cmd+Shift+5), tập 2 lần (1 lần tắt Wi-Fi)

## Rủi ro và thứ tự cắt phạm vi

| Rủi ro | Dấu hiệu sớm | Xử lý |
|---|---|---|
| Colab hết quota / ngắt liên tục | Pilot cho thấy 1 run > 90 phút | Chạy đêm; T5 W2 còn ≥ 5/9 run → Colab Pro; cuối cùng mới bỏ seed 2 |
| Tải dữ liệu lâu | Chưa xong Bước 0 trước W1 | `curl -C -` qua đêm; Colab tải thẳng vào Drive song song |
| Rare slice < 150 ảnh | `make splits` in số nhỏ | Dùng mức fallback đã khóa, ghi báo cáo — không tự đặt luật mới |
| SigLIP quá chậm | `--limit 200` cho thấy > 10 giờ | Bỏ SigLIP, bảng 2×2 thành 1×2, ghi hạn chế |
| App chậm / thiếu RAM | Khởi động > 60 s | Chỉ để 2 chỉ mục ViT-B-32 trong `index/` khi demo |
| Không nhờ được người chấm / timing | Cuối W2 chưa hẹn ai | Tự chấm qua `judge.py` (vẫn mù cấu hình), ghi thiên lệch |
| Tool không thắng random | Kết quả Bước 6 | Vẫn hợp lệ; so với arm_meta; không chạy thêm biến thể |

**Thứ tự cắt khi trễ:** CVAT (đã bỏ) → SigLIP → kappa (người chấm thứ hai) → chế độ timing trong app (thay bằng Sheet ghi image_id + giờ) → seed 2.
**Không bao giờ cắt:** khóa lock-v1, chấm bằng last.pt, báo cáo đủ mọi run.
