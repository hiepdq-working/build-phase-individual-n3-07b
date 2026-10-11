# N3-07B · Data Engine: tìm ca hiếm bằng CLIP và chứng minh chúng giúp detector

Dự án tìm **cảnh lái xe hiếm** (người đi bộ ban đêm dưới mưa/tuyết) trong kho 40.000 ảnh [BDD100K](https://www.vis.xyz/bdd100k/) chưa dùng nhãn. Có thể tìm bằng câu tiếng Anh hoặc bằng ảnh mẫu (CLIP, nguyên ảnh + cắt ô). Ảnh tìm được được xếp hạng theo **độ bất định** của detector. Thí nghiệm có đối chứng (3 arm × 3 seed) kiểm tra xem 500 ảnh do công cụ chọn có giúp YOLOv8s phát hiện người đi bộ tốt hơn 500 ảnh chọn ngẫu nhiên không.

> Phiên bản **làm một mình**: giữ đủ 3 con số + 1 demo của *Kế hoạch chi tiết* và các yêu cầu **Must** của *SRD v0.3*. Chỉ bỏ phần phối hợp nhóm và các mục *Could*. Xem [docs/plan.md](docs/plan.md).

## 3 con số + 1 demo

| # | Kết quả | Đo bằng | So với | File |
|---|---|---|---|---|
| 1 | Tìm tốt hơn với vật nhỏ, cảnh tối | Precision@50 trên 20 truy vấn tự động + 5 truy vấn chấm mù | CLIP nguyên ảnh (`whole`) so với nguyên ảnh + cắt ô (`whole_tiles`) | `results/search/p_at_50_auto_table.csv`, `results/relevance/p_at_50_human.csv` |
| 2 | Tìm ca hiếm nhanh hơn | Số ca hiếm *verified* trong 15 phút | Duyệt tay | `results/timing/summary.csv` |
| 3 | Dữ liệu tìm được giúp model tốt hơn | AP50 pedestrian trên rare slice, mean ± std qua 3 seed | `arm_random` (kèm `arm_meta`) | `results/detector_scores.csv`, `report/figures/arms_chart.png` |
| 4 | Demo offline trên Mac | Tìm bằng câu/ảnh, lọc metadata, uncertainty, so sánh whole/tiles, xuất CSV | | `app/app.py` |

Kết quả 3 không tăng **vẫn là kết quả hợp lệ** nếu thí nghiệm đúng quy trình và được báo cáo trung thực. Luật kết luận đã được khóa trước trong `configs/decisions.yaml`.

## Pipeline

```
G0  BDD100K ─► images/boxes.parquet ─► nhãn YOLO ─► splits: base 5k │ pool 40k │ dev 1k │ rare_val │ val2k
G1–G3  pool ─► 7 crop/ảnh (nguyên + lưới 3×2, chồng 20%, pad_square) ─► CLIP ─► FAISS: vitb32_whole, vitb32_whole_tiles
G4–G6  câu / ảnh ─► điểm ảnh = max(tile, prompt) ─► P@50 (20 câu tự động + 5 câu chấm mù)
G7     YOLOv8s (COCO) train trên base ─► detector gốc
G8     detector gốc trên pool ─► uncertainty = #box conf∈[0.2,0.5] / #box conf≥0.2
G9     arm_tool = 1.500 ảnh CLIP cao nhất → 500 ảnh uncertainty cao nhất │ arm_random │ arm_meta
G10    9 run = 3 arm × seed {0,1,2}, mỗi run train trên base ∪ arm (5.500 ảnh)
G11    chấm last.pt: AP50 pedestrian (rare_val), mAP50 (val2k) ─► mean ± std, tool − random, kết luận
```

| Việc | Chạy ở đâu |
|---|---|
| Metadata, nhãn YOLO, splits, chọn arm, P@50, tổng hợp số, app, báo cáo | **Mac** (VS Code, PyTorch CPU/MPS) |
| Embedding 280.000 crop, train YOLO 10 lần, detector trên 40.000 ảnh, chấm AP50 | **Google Colab** (T4 GPU) |
| Trung chuyển ảnh, nhãn, chỉ mục, checkpoint | **Google Drive** `MyDrive/n3-07b/` |
| Code, splits, config, CSV kết quả | **GitHub** (repo này) |

## Cấu trúc repo

```
n3-07b/
├── configs/
│   ├── decisions.yaml        # bảng quyết định khóa trước (tag lock-v1) — code chỉ đọc số từ đây
│   ├── queries.yaml          # 4 prompt tool, 20 eval_auto, 5 eval_human
│   └── main_model.yaml       # (tạo ở Bước 5.3) model chính chọn từ bảng P@50
├── splits/                   # base, pool, dev, rare_val, val2k, arm_* (.txt) + registry.csv (SHA-256)
├── src/
│   ├── common/               # paths.py (biến môi trường), config.py
│   ├── data/                 # build_meta, bdd_to_yolo, make_splits, check_splits, splits_io
│   ├── search/               # tiling, embed (Colab), check_index, engine
│   ├── uncertainty/          # score, run_pool (Colab)
│   ├── select/               # tool_select, make_arms, arm_report
│   ├── train/                # datasets, train_run, eval_detector (Colab)
│   └── eval/                 # gt_rules, eval_search, export_blind, score_blind, verify_timing, aggregate
├── app/                      # app.py (demo + timing), judge.py (chấm mù)
├── notebooks/                # 00_setup_drive, 01_embed, 02_train (Colab)
├── scripts/                  # download_bdd, pack_labels, reproduce, freeze_checks
├── results/                  # CSV kết quả — CÓ commit (mọi con số truy được về đây)
├── report/                   # report.md, slides_outline.md, figures/
├── docs/                     # kế hoạch, nguồn dữ liệu, pilot, OI, truy vết SRD, guideline chấm
├── tests/                    # pytest (không cần torch/dữ liệu)
├── data/  index/  runs/      # KHÔNG commit (.gitignore)
└── Makefile                  # make help
```

## Bắt đầu

### Bước 0 — Môi trường (Mac, Python 3.11)

```bash
git clone https://github.com/<user>/n3-07b-solo.git ~/n3-07b && cd ~/n3-07b
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements-mac.txt
pip freeze > requirements-mac.lock.txt
python -c "import torch;print(torch.__version__, torch.backends.mps.is_available())"
pytest -q
```

### Bước 0.3 — Tải BDD100K

Đăng ký tài khoản miễn phí tại <https://bdd-data.berkeley.edu/>. Nếu trang chậm, dùng mirror chính thức <https://dl.cv.ethz.ch/bdd100k/data/>. Không dùng bản tải lại trên Kaggle hay Hugging Face.

```bash
bash scripts/download_bdd.sh      # 3 file zip → data/, in số ảnh (70.000 / 10.000) và ghi docs/bdd100k.sha256
```

Sau đó chạy `notebooks/00_setup_drive.ipynb` trên Colab để có bản dữ liệu trên Drive.

### Các bước tiếp theo

Mỗi bước có phần **Kiểm tra** lấy từ SRD. Chỉ sang bước sau khi phần Kiểm tra của bước trước đã qua.

| Bước | Lệnh chính | Nơi chạy |
|---|---|---|
| 1 · G0 dữ liệu | `make meta yolo splits check` → `make pack` → kéo zip lên Drive | Mac |
| 2 · G1–G3 chỉ mục | `notebooks/01_embed.ipynb` → tải `index/vitb32_*` về Mac → `python -m src.search.check_index vitb32_whole_tiles` | Colab → Mac |
| 3 · G4–G6 P@50 | commit `queries.yaml` + `gt_rules.py` **trước**, rồi `make eval-search` | Mac |
| 4 · G7 pilot, lock-v1, base | `02_train.ipynb` (pilot) → chốt `train.epochs` → `git tag lock-v1` → `RUN_ID="base"` | Colab |
| 5 · G8–G9 arm | `02_train.ipynb` ô 8 → `configs/main_model.yaml` → `make tool-select arms arm-report` → commit **trước** run arm đầu tiên | Colab → Mac |
| 6 · G10–G11 9 run | `02_train.ipynb` với 9 `RUN_ID` → `make aggregate` | Colab → Mac |
| 7 · App | `make app` | Mac |
| 8 · Chấm mù + timing | `make blind judge score-blind`, `make app` (chế độ Timing) → `make timing` | Mac + 2 bạn |
| 9 · Freeze | `make reproduce freeze-checks` → `git tag freeze` | Mac |

Chi tiết từng bước nằm trong tài liệu *N3-07B làm một mình — checklist từng bước*. Lịch tuần nằm ở [docs/plan.md](docs/plan.md).

## Chạy demo

```bash
source .venv/bin/activate
pip install -r requirements-mac.lock.txt
streamlit run app/app.py           # http://localhost:8501 — chạy được khi tắt Wi-Fi
```

App cần `data/meta/images.parquet`, `index/vitb32_whole{,_tiles}/` và (để hiện uncertainty) `results/uncertainty/base_pool.csv`.

## Toàn vẹn thí nghiệm

- `configs/decisions.yaml` được khóa bằng tag `lock-v1` trước run đầu tiên. CI kiểm `git diff lock-v1 -- configs/decisions.yaml` phải rỗng (NFR-01).
- Prompt tool, 20 truy vấn eval và `gt_rules.py` được commit trước lần tìm đầu tiên (NFR-04). 3 arm được commit trước run arm đầu tiên (FR-SEL-08).
- Không chọn prompt, ngưỡng, hyperparameter hay checkpoint dựa trên `rare_val`/`val2k` (NFR-03). Luôn chấm bằng `last.pt`.
- Báo cáo đủ 9 run và điểm từng seed, kể cả kết quả bất lợi (NFR-11).
- `scripts/reproduce.sh` tái tạo 3 con số từ file đã commit (FR-EVL-19).

## Dữ liệu và giấy phép

- Code dùng giấy phép [MIT](LICENSE).
- **BDD100K chỉ dùng cho nghiên cứu và không được phân phối lại.** Ảnh, embedding, chỉ mục và weights **không** nằm trong repo: `.gitignore` chặn chúng, và CI báo lỗi nếu có file nào lọt vào (NFR-39, 40). Drive chỉ chia sẻ với người chấm, không có link công khai.
- Báo cáo và slide ghi nguồn (NFR-41):
  > Yu et al., *BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning*, CVPR 2020.
- CLIP: [OpenCLIP](https://github.com/mlfoundations/open_clip) (ViT-B-32 LAION-2B; ViT-B-16-SigLIP-384 WebLI). Detector: [Ultralytics YOLOv8s](https://github.com/ultralytics/ultralytics), khởi tạo từ trọng số COCO.
- Tracker, log và CSV chỉ dùng alias (`acc1`, `p1`, `j1`), không ghi email (NFR-42).
