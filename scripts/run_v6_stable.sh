#!/bin/bash
# v6 stability study: does the chi2-gate fix and/or the map filter remove the failure runs?
#
# Variants (each = DSLAM_MODE + DSLAM_CHI2_GATE + DSLAM_MAP_FILTER):
#   <mode>_w_mf0   v4/v5 behaviour (weighted chi2 gate, no map filter)  -> reference
#   <mode>_r_mf0   raw chi2 gate only
#   <mode>_w_mf1   map filter only
#   <mode>_r_mf1   both fixes
#
# Usage (from ~/ORB_SLAM3):
#   ./scripts/run_v6_stable.sh                                   # everything (~2-3 h)
#   SEQS="walking_xyz" RUNS=10 ./scripts/run_v6_stable.sh         # start here (~45 min)
#   MODES="geo" GATES="raw" FILTERS="0 1" ./scripts/run_v6_stable.sh
#
# Output: results_v6/<seq>/<variant>_<run>.{txt,eval,log} and a summary at the end.

SEQS="${SEQS:-walking_xyz walking_static sitting_static}"
MODES="${MODES:-geo proposed}"
GATES="${GATES:-weighted raw}"
FILTERS="${FILTERS:-0 1}"
RUNS="${RUNS:-10}"
FAIL_TH="${FAIL_TH:-0.1}"          # ATE RMSE above this (m) counts as a failed run
CALIB=Examples/RGB-D/TUM3.yaml
ROOT="${ROOT:-$HOME/ORB_SLAM3/results_v6}"

for SEQ in $SEQS; do
  DATASET=~/datasets/rgbd_dataset_freiburg3_$SEQ
  ASSOC=$DATASET/association_new.txt
  GT=$DATASET/groundtruth.txt
  OUT=$ROOT/$SEQ
  mkdir -p "$OUT"
  for MODE in $MODES; do
    for GATE in $GATES; do
      for MF in $FILTERS; do
        V="${MODE}_${GATE:0:1}_mf${MF}"
        for i in $(seq 1 "$RUNS"); do
          echo "[$(date +%H:%M)] $SEQ  $V  run $i"
          DSLAM_MODE=$MODE DSLAM_CHI2_GATE=$GATE DSLAM_MAP_FILTER=$MF \
            ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt $CALIB $DATASET $ASSOC > "$OUT/${V}_$i.log" 2>&1
          mv CameraTrajectory.txt "$OUT/${V}_$i.txt"
          evo_ape tum $GT "$OUT/${V}_$i.txt" -a > "$OUT/${V}_$i.eval"
        done
      done
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
