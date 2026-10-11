PY ?= .venv/bin/python

.PHONY: help setup lock check-env meta yolo splits check test lint pack app judge \
        eval-search arms tool-select arm-report blind score-blind timing aggregate reproduce freeze-checks

help:            ## Liệt kê các lệnh
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-14s %s\n", $$1, $$2}'

setup:           ## Bước 0.1 — tạo .venv và cài thư viện Mac
	python3.11 -m venv .venv && $(PY) -m pip install -U pip && $(PY) -m pip install -r requirements-mac.txt

lock:            ## Ghi requirements-mac.lock.txt (NFR-15)
	$(PY) -m pip freeze > requirements-mac.lock.txt

check-env:       ## In phiên bản torch và MPS
	$(PY) -c "import torch;print(torch.__version__, 'MPS:', torch.backends.mps.is_available())"

meta:            ## Bước 1.2 — images.parquet + boxes.parquet
	$(PY) -m src.data.build_meta

yolo:            ## Bước 1.3 — nhãn YOLO
	$(PY) -m src.data.bdd_to_yolo

splits:          ## Bước 1.4 — 5 split cố định
	$(PY) -m src.data.make_splits

check:           ## Kiểm ràng buộc split (chạy trước mỗi lần train)
	$(PY) -m src.data.check_splits

pack:            ## Bước 1.6 — nén nhãn YOLO + meta để đưa lên Drive
	bash scripts/pack_labels.sh

eval-search:     ## Bước 3.4 — P@50 tự động whole vs tiles (con số 1)
	$(PY) -m src.eval.eval_search vitb32_whole vitb32_whole_tiles

tool-select:     ## Bước 5.4 — chọn arm_tool
	$(PY) -m src.select.tool_select

arms:            ## Bước 5.5 — arm_random + arm_meta, rồi check_splits
	$(PY) -m src.select.make_arms && $(PY) -m src.data.check_splits

arm-report:      ## Bước 5.6 — bảng thành phần 3 arm
	$(PY) -m src.select.arm_report

aggregate:       ## Bước 6.2 — tổng hợp 9 run + biểu đồ (con số 3)
	$(PY) -m src.eval.aggregate

app:             ## Bước 7 — demo Streamlit
	.venv/bin/streamlit run app/app.py

blind:           ## Bước 8.1 — xuất bộ chấm mù
	$(PY) -m src.eval.export_blind vitb32_whole vitb32_whole_tiles

judge:           ## Bước 8.2 — trang chấm mù
	.venv/bin/streamlit run app/judge.py

score-blind:     ## Bước 8.3 — giải mù, P@50 tay, kappa
	$(PY) -m src.eval.score_blind

timing:          ## Bước 8.4 — xác minh claim timing (con số 2)
	$(PY) -m src.eval.verify_timing

reproduce:       ## Bước 9.1 — tái tạo 3 con số
	bash scripts/reproduce.sh

freeze-checks:   ## Bước 9.2 — cổng kiểm tra trước freeze
	bash scripts/freeze_checks.sh

test:            ## pytest
	$(PY) -m pytest -q

lint:            ## ruff
	$(PY) -m ruff check .
