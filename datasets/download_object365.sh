#!/usr/bin/env bash
set -euo pipefail

sudo apt install aria2 bc -y

export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-./datasets}"
echo "DETECTRON2_DATASETS=${DETECTRON2_DATASETS}"

DATASET_ROOT="${DETECTRON2_DATASETS}/Objects365/val"
BASE="https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365数据集/val"

# Tune these values for your connection and the download server.
PARALLEL_DOWNLOADS=4
CONNECTIONS_PER_FILE=4
SPLITS_PER_FILE=4

if ! command -v aria2c >/dev/null 2>&1; then
    echo "Error: aria2c is not installed."
    echo "Ubuntu/Debian: sudo apt install aria2"
    echo "Fedora:         sudo dnf install aria2"
    exit 1
fi

echo "Dataset will be downloaded to: ${DATASET_ROOT}"
mkdir -p "${DATASET_ROOT}/archives" "${DATASET_ROOT}/images"
# cd "${DATASET_ROOT}"
echo "Current working directory: $(pwd)"

echo "Downloading validation annotations..."
aria2c \
    --continue=true \
    --allow-overwrite=true \
    --auto-file-renaming=false \
    --dir="${DATASET_ROOT}" \
    --out="zhiyuan_objv2_val.json" \
    --max-tries=10 \
    --retry-wait=5 \
    --timeout=60 \
    --summary-interval=10 \
    "${BASE}/zhiyuan_objv2_val.json"

DOWNLOAD_LIST="$(mktemp)"
trap 'rm -f "${DOWNLOAD_LIST}"' EXIT

# Validation patches 0 to 15 are stored under v1.
for i in $(seq 0 15); do
    echo "${BASE}/images/v1/patch${i}.tar.gz" >> "${DOWNLOAD_LIST}"
done

# Validation patches 16 to 43 are stored under v2.
for i in $(seq 16 43); do
    echo "${BASE}/images/v2/patch${i}.tar.gz" >> "${DOWNLOAD_LIST}"
done

echo "Downloading Objects365 validation patches in parallel..."
echo "Parallel downloads: ${PARALLEL_DOWNLOADS}"
echo "Connections per patch: ${CONNECTIONS_PER_FILE}"
echo "Splits per patch: ${SPLITS_PER_FILE}"

aria2c \
    --input-file="${DOWNLOAD_LIST}" \
    --dir="${DATASET_ROOT}/archives" \
    --continue=true \
    --allow-overwrite=true \
    --auto-file-renaming=false \
    --max-concurrent-downloads="${PARALLEL_DOWNLOADS}" \
    --max-connection-per-server="${CONNECTIONS_PER_FILE}" \
    --split="${SPLITS_PER_FILE}" \
    --min-split-size=10M \
    --file-allocation=none \
    --max-tries=10 \
    --retry-wait=5 \
    --timeout=60 \
    --summary-interval=10 \
    --console-log-level=info

echo "Extracting validation image archives..."

shopt -s nullglob
archives=(${DATASET_ROOT}/archives/*.tar.gz)

if (( ${#archives[@]} == 0 )); then
    echo "Error: no archive files found in ${DATASET_ROOT}/archives"
    exit 1
fi

for archive in "${archives[@]}"; do
    echo "Extracting: ${archive}"
    tar -xzf "${archive}" -C "${DATASET_ROOT}/images/"
done

echo "Objects365 validation set download and extraction completed."
