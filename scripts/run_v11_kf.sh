#!/bin/bash
# v11: the final fix should touch ONLY the keyframe decision, and YOLO should decide which outliers to ignore.
# v10 (walking_xyz, 5 runs): ignoring outlier labels in the keyframe decision (E4 + E5) is enough:
# 0.032 m, 0/5 failures, 50 keyframes instead of 358. E5 also fed the tracking-OK test, so it is split:
#   E5a = inlier count for the tracking-OK test, E5b = inlier count for the keyframe decision.
#
#   A0_orbslam3   reference: plain ORB-SLAM3
#   E45           v10 fix (E4 + E5a + E5b)
#   K_kf          keyframe decision only (E4 + E5b)        <- clean geometric fix
#   K_track       tracking-OK test only (E5a)              <- sanity check, expect no fix
#   S_kf          K_kf, but only outliers inside YOLO boxes are ignored  <- semantic fix
#   S_kf_mf       S_kf + no new map points inside YOLO boxes (DSLAM_MAP_FILTER=1)
#
# All variants: DSLAM_MODE=baseline (no weights), raw gate, Huber, normal labels.
# Usage (from ~/ORB_SLAM3):
#   ./scripts/run_v11_kf.sh                                         (walking_xyz, 5 runs, ~25 min)
#   SEQS="walking_static sitting_static walking_halfsphere walking_rpy long_office_household" ./scripts/run_v11_kf.sh

SEQS="${SEQS:-walking_xyz}"
RUNS="${RUNS:-5}"
FAIL_TH="${FAIL_TH:-0.1}"
CALIB=Examples/RGB-D/TUM3.yaml
ROOT="${ROOT:-$HOME/ORB_SLAM3/results_v11}"

# name:labels:ignore   (all: DSLAM_MODE=baseline, gate raw, Huber)
# name:ignore:kf_semantic:map_filter
VARIANTS="
A0_orbslam3::0:0
E45:E4,E5:0:0
K_kf:E4,E5b:0:0
K_track:E5a:0:0
S_kf:E4,E5b:1:0
S_kf_mf:E4,E5b:1:1
"

for SEQ in $SEQS; do
  DATASET=~/datasets/rgbd_dataset_freiburg3_$SEQ
  ASSOC=$DATASET/association_new.txt
  GT=$DATASET/groundtruth.txt
  OUT=$ROOT/$SEQ
  mkdir -p "$OUT"
  for VDEF in $VARIANTS; do
    IFS=: read -r V IGN SEM MF <<< "$VDEF"
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" $V "* ]]; then continue; fi
    for i in $(seq 1 "$RUNS"); do
      echo "[$(date +%H:%M)] $SEQ  $V  run $i"
      DSLAM_MODE=baseline DSLAM_CHI2_GATE=raw DSLAM_LABELS=opt DSLAM_IGNORE=$IGN DSLAM_KF_SEMANTIC=$SEM DSLAM_KERNEL=huber DSLAM_MAP_FILTER=$MF \
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
