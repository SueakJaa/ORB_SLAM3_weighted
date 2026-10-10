#!/bin/bash
SEQS="${SEQS:-walking_xyz walking_static sitting_static}"
MODES="${MODES:-baseline geo yolo proposed}"
RUNS="${RUNS:-10}"
CALIB=Examples/RGB-D/TUM3.yaml       # same as paper (correct for fr3 is TUM3.yaml)

for SEQ in $SEQS; do
  DATASET=~/datasets/rgbd_dataset_freiburg3_$SEQ
  ASSOC=$DATASET/association_new.txt
  GT=$DATASET/groundtruth.txt
  OUT=~/ORB_SLAM3/results_v5_gpu/$SEQ
  mkdir -p $OUT
  for MODE in $MODES; do
    for i in $(seq 1 $RUNS); do
      echo "[$(date +%H:%M)] $SEQ  $MODE  run $i"
      DSLAM_MODE=$MODE ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt $CALIB $DATASET $ASSOC > $OUT/${MODE}_$i.log 2>&1
      mv CameraTrajectory.txt $OUT/${MODE}_$i.txt
      evo_ape tum $GT $OUT/${MODE}_$i.txt -a > $OUT/${MODE}_$i.eval
    done
  done
done

echo; echo "===== SUMMARY (ATE RMSE, m) ====="
for SEQ in $SEQS; do for MODE in $MODES; do
  F=$(ls ~/ORB_SLAM3/results_v5_gpu/$SEQ/${MODE}_*.eval 2>/dev/null)
  [ -z "$F" ] && continue
  grep -h rmse $F | awk -v s=$SEQ -v m=$MODE '{x[NR]=$2; t+=$2} END {mu=t/NR; for(i=1;i<=NR;i++) v+=(x[i]-mu)^2; sd=(NR>1)?sqrt(v/(NR-1)):0; printf "%-15s %-9s n=%d  mean=%.4f  std=%.4f\n", s, m, NR, mu, sd}'
done; done
