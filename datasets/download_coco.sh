#!/bin/bash
# download_coco2017_fast.sh - MAX 16-chunk parallel download (aria2c limit)
# sudo dnf install aria2
# Usage: chmod +x download_coco2017_fast.sh && ./download_coco2017_fast.sh

set -e

apt install aria2 bc -y

start_time=$(date +%s.%N)

BASE_URL="http://images.cocodataset.org"
export DETECTRON2_DATASETS=${DETECTRON2_DATASETS:-./datasets}
echo "DETECTRON2_DATASETS=$DETECTRON2_DATASETS"

DATASET_ROOT="$DETECTRON2_DATASETS/coco"
mkdir -p $DATASET_ROOT/{images/train2017,images/val2017,annotations}

download_parallel() {
    local url=$1
    local output_dir=$2
    
    aria2c --no-netrc --max-connection-per-server=16 --max-concurrent-downloads=16 --split=16 --min-split-size=10M -c -x 16 -s 16 -d "$output_dir" "$url" --summary-interval=60
}

echo "Downloading train2017.zip (~19GB) with MAX 16 parallel chunks..."
download_parallel "$BASE_URL/zips/train2017.zip" $DATASET_ROOT/images/
unzip -q -d $DATASET_ROOT/images/ $DATASET_ROOT/images/train2017.zip && rm $DATASET_ROOT/images/train2017.zip

echo "Downloading val2017.zip (~1GB) with 16 parallel chunks..."
download_parallel "$BASE_URL/zips/val2017.zip" $DATASET_ROOT/images/
unzip -q -d $DATASET_ROOT/images/ $DATASET_ROOT/images/val2017.zip && rm $DATASET_ROOT/images/val2017.zip

echo "Downloading annotations (~241MB) with 16 parallel chunks..."
download_parallel "$BASE_URL/annotations/annotations_trainval2017.zip" $DATASET_ROOT/
unzip -q -d $DATASET_ROOT/ $DATASET_ROOT/annotations_trainval2017.zip && rm $DATASET_ROOT/annotations_trainval2017.zip

echo "Done! Downloads complete."
ls -lh $DATASET_ROOT/images/train2017/ | head -5

end_time=$(date +%s.%N)
duration=$(echo "$end_time - $start_time" | bc -l)
echo "Total processing time: ${duration} seconds (~$(printf "%.1f" $(echo "$duration / 60" | bc -l)) minutes)"