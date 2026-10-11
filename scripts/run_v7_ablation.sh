#!/bin/bash
# v7 ablation study: WHY does the method work?
# v6 showed: with ORB-SLAM3's normal (raw) outlier test the weights change nothing (0.72 m = baseline);
# with the weighted test (= in practice no rejection) error drops to ~0.03 m.
# Each variant below answers one question.
#
#   name               mode      gate      kernel  map  question it answers
#   A0_orbslam3        baseline  raw       huber   0    reference: plain ORB-SLAM3 (expect ~0.72)
#   A1_late_only       baseline  late      huber   0    is delaying hard rejection alone enough? (no weights)
#   B1_cauchy          baseline  raw       cauchy  0    does a standard heavier-tailed kernel fix it?
#   B2_welsch          baseline  raw       welsch  0    Welsch kernel = w_geo recomputed every iteration
#   B3_welsch_late     baseline  late      welsch  0    Welsch + delayed rejection
#   C0_geo_v5          geo       weighted  huber   0    current method (expect ~0.03, 2/10 failures)
#   C1_geo_late        geo       late      huber   0    clean version: soft first, hard test at the end
#   C2_proposed_late   proposed  late      huber   0    C1 + YOLO semantic weights
#
# Usage (from ~/ORB_SLAM3):
#   ./scripts/run_v7_ablation.sh                                  # walking_xyz, all 8 variants, 10 runs (~50 min)
#   ONLY="A1_late_only C1_geo_late" RUNS=5 ./scripts/run_v7_ablation.sh
#   SEQS="walking_xyz walking_static sitting_static" ./scripts/run_v7_ablation.sh

SEQS="${SEQS:-walking_xyz}"
RUNS="${RUNS:-10}"
FAIL_TH="${FAIL_TH:-0.1}"
CALIB=Examples/RGB-D/TUM3.yaml
ROOT="${ROOT:-$HOME/ORB_SLAM3/results_v7}"

VARIANTS="
A0_orbslam3:baseline:raw:huber:0
A1_late_only:baseline:late:huber:0
B1_cauchy:baseline:raw:cauchy:0
B2_welsch:baseline:raw:welsch:0
B3_welsch_late:baseline:late:welsch:0
C0_geo_v5:geo:weighted:huber:0
C1_geo_late:geo:late:huber:0
C2_proposed_late:proposed:late:huber:0
"

for SEQ in $SEQS; do
  DATASET=~/datasets/rgbd_dataset_freiburg3_$SEQ
  ASSOC=$DATASET/association_new.txt
  GT=$DATASET/groundtruth.txt
  OUT=$ROOT/$SEQ
  mkdir -p "$OUT"
  for VDEF in $VARIANTS; do
    IFS=: read -r V MODE GATE KERNEL MF <<< "$VDEF"
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" $V "* ]]; then continue; fi
    for i in $(seq 1 "$RUNS"); do
      echo "[$(date +%H:%M)] $SEQ  $V  run $i"
      DSLAM_MODE=$MODE DSLAM_CHI2_GATE=$GATE DSLAM_KERNEL=$KERNEL DSLAM_MAP_FILTER=$MF \
        ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt $CALIB $DATASET $ASSOC > "$OUT/${V}_$i.log" 2>&1
      mv CameraTrajectory.txt "$OUT/${V}_$i.txt"
      evo_ape tum $GT "$OUT/${V}_$i.txt" -a > "$OUT/${V}_$i.eval"
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
