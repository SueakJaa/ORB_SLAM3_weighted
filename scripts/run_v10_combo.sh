#!/bin/bash
# v10: the failure needs SEVERAL label uses together. Which combination?
# v9 (5 runs each, walking_xyz): no single use fixes it. E4 (keyframe decision) halves the error
# (0.36 m), E5 (inlier count) helps a little (0.60 m), E1/E2/E3 alone do nothing.
# E2 alone could not work: ORBmatcher::SearchByProjection also skips last-frame outliers -> new E6.
# This run also saves KeyFrameTrajectory.txt per run and reports the mean keyframe count,
# to test the idea that outlier labels on people trigger extra keyframes.
#
#   A0_orbslam3   reference: plain ORB-SLAM3
#   D2_nolabels   reference: no outlier labels at all
#   E2E6          keep outlier links AND keep matching them frame-to-frame
#   E45           keyframe decision + inlier count (both feed NeedNewKeyFrame)
#   E2456         both groups together
#   E_all         all six uses switched off -> should match D2 if no use is missing
#
# Usage (from ~/ORB_SLAM3):  ./scripts/run_v10_combo.sh          (5 runs each, ~25 min)

SEQS="${SEQS:-walking_xyz}"
RUNS="${RUNS:-5}"
FAIL_TH="${FAIL_TH:-0.1}"
CALIB=Examples/RGB-D/TUM3.yaml
ROOT="${ROOT:-$HOME/ORB_SLAM3/results_v10}"

# name:labels:ignore   (all: DSLAM_MODE=baseline, gate raw, Huber)
VARIANTS="
A0_orbslam3:opt:
D2_nolabels:none:
E2E6:opt:E2,E6
E45:opt:E4,E5
E2456:opt:E2,E4,E5,E6
E_all:opt:E1,E2,E3,E4,E5,E6
"

for SEQ in $SEQS; do
  DATASET=~/datasets/rgbd_dataset_freiburg3_$SEQ
  ASSOC=$DATASET/association_new.txt
  GT=$DATASET/groundtruth.txt
  OUT=$ROOT/$SEQ
  mkdir -p "$OUT"
  for VDEF in $VARIANTS; do
    IFS=: read -r V LABELS IGN <<< "$VDEF"
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" $V "* ]]; then continue; fi
    for i in $(seq 1 "$RUNS"); do
      echo "[$(date +%H:%M)] $SEQ  $V  run $i"
      DSLAM_MODE=baseline DSLAM_CHI2_GATE=raw DSLAM_LABELS=$LABELS DSLAM_IGNORE=$IGN DSLAM_KERNEL=huber DSLAM_MAP_FILTER=0 \
        ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt $CALIB $DATASET $ASSOC > "$OUT/${V}_$i.log" 2>&1
      if [ -f CameraTrajectory.txt ]; then
        mv CameraTrajectory.txt "$OUT/${V}_$i.txt"
        [ -f KeyFrameTrajectory.txt ] && mv KeyFrameTrajectory.txt "$OUT/${V}_${i}_kf.txt"
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

echo; echo "===== KEYFRAMES (mean number of keyframes in the final map per run) ====="
for SEQ in $SEQS; do
  for F in $(ls $ROOT/$SEQ/*_1.eval 2>/dev/null); do
    V=$(basename "$F" _1.eval)
    for K in $ROOT/$SEQ/${V}_*_kf.txt; do [ -f "$K" ] && wc -l < "$K"; done | \
      awk -v s=$SEQ -v v=$V '{t+=$1; n++} END {if(n) printf "%-15s %-16s keyframes mean=%.1f (n=%d)\n", s, v, t/n, n}'
  done
done
