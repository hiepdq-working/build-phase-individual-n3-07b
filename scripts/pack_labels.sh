#!/usr/bin/env bash
# Bước 1.6 — nén nhãn YOLO + metadata để kéo lên Drive (MyDrive/n3-07b/raw/), ghi checksum (FR-DAT-10).
set -euo pipefail
cd data/bdd100k && zip -qr ../yolo_labels.zip labels/100k && cd ..
zip -qr meta.zip meta
shasum -a 256 yolo_labels.zip meta.zip | tee -a ../docs/bdd100k.sha256
echo "Kéo data/yolo_labels.zip và data/meta.zip vào MyDrive/n3-07b/raw/"
