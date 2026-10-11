#!/usr/bin/env bash
# Bước 0.3 — tải BDD100K về Mac (chạy từ thư mục gốc repo). -C - để tải tiếp nếu đứt mạng.
# Cần đăng ký tài khoản tại https://bdd-data.berkeley.edu/ ; mirror chính thức: https://dl.cv.ethz.ch/bdd100k/data/
set -euo pipefail
MIRROR=${MIRROR:-https://dl.cv.ethz.ch/bdd100k/data}
mkdir -p data/raw docs
cd data/raw
for f in 100k_images_train.zip 100k_images_val.zip bdd100k_det_20_labels_trainval.zip; do
  curl -C - -fO "$MIRROR/$f"
done
shasum -a 256 *.zip | tee ../../docs/bdd100k.sha256
cd ..
for f in raw/*.zip; do unzip -q -n "$f" -d .; done
echo "train: $(ls bdd100k/images/100k/train | wc -l)  (kỳ vọng 70000)"
echo "val:   $(ls bdd100k/images/100k/val | wc -l)  (kỳ vọng 10000)"
ls bdd100k/labels/det_20/
