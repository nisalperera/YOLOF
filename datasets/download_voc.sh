#!/usr/bin/env bash
set -e

apt install aria2 bc -y

start_time=$(date +%s.%N)

export DETECTRON2_DATASETS=${DETECTRON2_DATASETS:-./datasets}
echo "DETECTRON2_DATASETS=$DETECTRON2_DATASETS"

BASE_DIR="$DETECTRON2_DATASETS/pascal-voc"
VOC2007_DIR="$BASE_DIR/2007"
VOC2012_DIR="$BASE_DIR/2012"

mkdir -p "$VOC2007_DIR" "$VOC2012_DIR"

download_and_extract() {
  local url="$1"
  local outdir="$2"
  local file
  file="$(basename "$url")"
  local archive="$outdir/$file"

  if [ ! -f "$archive" ]; then
    echo "Downloading $file ..."
    aria2c -c -x 16 -s 16 -k 1M --dir="$outdir" --out="$file" "$url"
  else
    echo "Already downloaded: $archive"
  fi

  echo "Extracting $file to $outdir..."
  tar -xf "$archive" -C "$outdir"
}

# VOC 2007
download_and_extract "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar" "$VOC2007_DIR"
download_and_extract "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar" "$VOC2007_DIR"
download_and_extract "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar" "$VOC2007_DIR"

# VOC 2012
download_and_extract "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar" "$VOC2012_DIR"
download_and_extract "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar" "$VOC2012_DIR"

echo "Done."

echo "Sample files from VOC 2007:"
ls -lh $VOC2007_DIR | head -5

echo "Sample files from VOC 2012:"
ls -lh $VOC2012_DIR | head -5

end_time=$(date +%s.%N)
duration=$(echo "$end_time - $start_time" | bc -l)
echo "Total processing time: ${duration} seconds (~$(printf "%.1f" $(echo "$duration / 60" | bc -l)) minutes)"