#!/bin/bash
# v8: is the gain in the pose optimisation, or in how ORB-SLAM3 USES the outlier labels afterwards?
# v7 showed: every variant that ever marks points as outliers fails (~0.6-0.7 m);
# only C0 (weighted gate = nothing is ever marked outlier) works (~0.03 m).
# DSLAM_LABELS changes only the labels handed to the rest of ORB-SLAM3, never the pose.
#
#   name                    mode      gate      labels  question
#   A0_orbslam3             baseline  raw       opt     reference: plain ORB-SLAM3
#   C0_geo_v5               geo       weighted  opt     reference: current method
#   D1_geo_v5_rawlabels     geo       weighted  raw     same pose maths as C0, normal labels. Fails => labels matter
#   D2_orbslam3_nolabels    baseline  raw       none    plain ORB-SLAM3 pose maths, no outlier labels. Works => labels are the whole story
#   D3_geo_raw_nolabels     geo       raw       none    weights + normal rejection inside, no labels outside
#
# Usage (from ~/ORB_SLAM3):  ./scripts/run_v8_labels.sh            (~30 min)
#                            ONLY="D1_geo_v5_rawlabels D2_orbslam3_nolabels" RUNS=5 ./scripts/run_v8_labels.sh

SEQS="${SEQS:-walking_xyz}"
RUNS="${RUNS:-10}"
FAIL_TH="${FAIL_TH:-0.1}"
CALIB=Examples/RGB-D/TUM3.yaml
ROOT="${ROOT:-$HOME/ORB_SLAM3/results_v8}"

VARIANTS="
A0_orbslam3:baseline:raw:opt
C0_geo_v5:geo:weighted:opt
D1_geo_v5_rawlabels:geo:weighted:raw
D2_orbslam3_nolabels:baseline:raw:none
D3_geo_raw_nolabels:geo:raw:none
"

for SEQ in $SEQS; do
  DATASET=~/datasets/rgbd_dataset_freiburg3_$SEQ
  ASSOC=$DATASET/association_new.txt
  GT=$DATASET/groundtruth.txt
  OUT=$ROOT/$SEQ
  mkdir -p "$OUT"
  for VDEF in $VARIANTS; do
    IFS=: read -r V MODE GATE LABELS <<< "$VDEF"
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" $V "* ]]; then continue; fi
    for i in $(seq 1 "$RUNS"); do
      echo "[$(date +%H:%M)] $SEQ  $V  run $i"
      DSLAM_MODE=$MODE DSLAM_CHI2_GATE=$GATE DSLAM_LABELS=$LABELS DSLAM_KERNEL=huber DSLAM_MAP_FILTER=0 \
        ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt $CALIB $DATASET $ASSOC > "$OUT/${V}_$i.log" 2>&1
      if [ -f CameraTrajectory.txt ]; then
        mv CameraTrajectory.txt "$OUT/${V}_$i.txt"
        evo_ape tum $GT "$OUT/${V}_$i.txt" -a > "$OUT/${V}_$i.eval"
      else
        echo "  !! run $i produced no trajectory (crash?) - see $OUT/${V}_$i.log"
      fi
    done
  done
done

echo; echo "===== SUMMARY (ATE RMSE, m; fail = run with RMSE > $FAIL_TH) ====="
for SEQ in $SEQS; do
  for F in $(ls $ROOT/$SEQ/*_1.eval 2>/dev/null); do
    V=$(basename "$F" _1.eval)
    # values sorted first so the median works with plain awk (Ubuntu's default mawk has no asort)
    grep -h rmse $ROOT/$SEQ/${V}_*.eval | awk '{print $2}' | sort -g | awk -v s=$SEQ -v v=$V -v th=$FAIL_TH '
      {x[NR]=$1; t+=$1; if($1>th) f++}
      END {mu=t/NR; for(i=1;i<=NR;i++) d+=(x[i]-mu)^2; sd=(NR>1)?sqrt(d/(NR-1)):0;
           med=(NR%2)?x[(NR+1)/2]:(x[NR/2]+x[NR/2+1])/2;
           printf "%-15s %-16s n=%-2d mean=%.4f  median=%.4f  std=%.4f  fail=%d/%d\n", s, v, NR, mu, med, sd, f+0, NR}'
  done
done
