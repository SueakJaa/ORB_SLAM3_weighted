#!/bin/bash
# Download and unpack the Bonn RGB-D Dynamic sequences used in configs/bonn_baseline.yaml.
# Usage (from ~/ORB_SLAM3):  bash scripts/eval/download_bonn.sh      (resumable; re-run if it stops)
# Pattern from https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/ : .../rgbd_dynamic2019/<name>.zip
DATA=${DATA:-$HOME/datasets}
SEQS=${SEQS:-"balloon balloon2 crowd crowd2 person_tracking person_tracking2 moving_nonobstructing_box moving_obstructing_box synchronous static"}
mkdir -p "$DATA" && cd "$DATA" || exit 1
command -v aria2c >/dev/null || { echo "installing aria2 (parallel, resumable downloads)"; sudo apt install -y aria2; }
for s in $SEQS; do
  n=rgbd_bonn_$s
  if [ -f "$n/groundtruth.txt" ]; then echo "ok    $n (already there)"; continue; fi
  aria2c -x 8 -s 8 -c -q --summary-interval=0 "https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/$n.zip" \
    && unzip -q -o "$n.zip" && rm -f "$n.zip" && echo "ok    $n" || echo "FAIL  $n (run the script again to resume)"
done
echo; for s in $SEQS; do n=rgbd_bonn_$s; [ -f "$n/groundtruth.txt" ] && echo "ready $n ($(ls $n/rgb | wc -l) images)" || echo "missing $n"; done
