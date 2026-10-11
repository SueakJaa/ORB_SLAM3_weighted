#!/bin/bash
# v9: WHICH use of the outlier label breaks ORB-SLAM3 in dynamic scenes?
# v8 showed: plain ORB-SLAM3 pose maths + no outlier labels (D2) = 0.025 m, 0/10 failures,
# while normal labels give ~0.69 m, 10/10 failures. ORB-SLAM3 uses the label in 5 places;
# each E-variant switches off exactly one of them (plain ORB-SLAM3 otherwise, no weights, no YOLO).
#
#   A0_orbslam3     reference: plain ORB-SLAM3
#   D2_nolabels     reference: no outlier labels at all (v8 winner)
#   E1_discard      outlier points keep their map link during tracking
#   E2_prekf        outlier links not removed before the keyframe / next frame
#   E3_found        outliers still count as "found" in map-point statistics
#   E4_kfclose      outliers count as tracked close points in the keyframe decision
#   E5_count        outliers count in the inlier number (tracking-OK test, keyframe ratio)
#   E12             E1 + E2 together (the two "keep the link" places)
#
# Usage (from ~/ORB_SLAM3):  ./scripts/run_v9_sites.sh                 (5 runs each, ~30 min)
#                            RUNS=10 ONLY="E1_discard E2_prekf" ./scripts/run_v9_sites.sh

SEQS="${SEQS:-walking_xyz}"
RUNS="${RUNS:-5}"
FAIL_TH="${FAIL_TH:-0.1}"
CALIB=Examples/RGB-D/TUM3.yaml
ROOT="${ROOT:-$HOME/ORB_SLAM3/results_v9}"

# name:labels:ignore   (all: DSLAM_MODE=baseline, gate raw, Huber)
VARIANTS="
A0_orbslam3:opt:
D2_nolabels:none:
E1_discard:opt:E1
E2_prekf:opt:E2
E3_found:opt:E3
E4_kfclose:opt:E4
E5_count:opt:E5
E12:opt:E1,E2
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
