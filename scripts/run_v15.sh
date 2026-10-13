#!/bin/bash
# v15: does YOLO matter? Test on FAST camera motion in STATIC rooms (no people).
# There ORB-SLAM3 genuinely needs frequent keyframes, so a fixed cap (G10) should hurt,
# while SW_G10 only caps when YOLO sees people and should behave like plain ORB-SLAM3.
#
#   A0_orbslam3   plain ORB-SLAM3
#   G10           at most one keyframe every 10 frames, always
#   SW_G10        same cap, only while YOLO sees people (the proposed method)
#
# Sequences: give the TUM name without the "rgbd_dataset_" prefix. The calibration file
# is picked from the name (freiburg1 -> TUM1.yaml, freiburg2 -> TUM2.yaml, freiburg3 -> TUM3.yaml).
# Short fr3 names from earlier scripts (e.g. walking_xyz) still work.
# association_new.txt is created automatically with evaluation/associate.py if missing.
#
# Usage (from ~/ORB_SLAM3):
#   ./scripts/run_v15.sh                                    (default: 4 fast fr1 sequences, 10 runs)
#   SEQS="freiburg1_desk freiburg3_walking_xyz" RUNS=5 ./scripts/run_v15.sh
#   ONLY="G10 SW_G10" ./scripts/run_v15.sh

SEQS="${SEQS:-freiburg1_desk freiburg1_desk2 freiburg1_room freiburg1_360}"
RUNS="${RUNS:-10}"
FAIL_TH="${FAIL_TH:-0.1}"
ROOT="${ROOT:-$HOME/ORB_SLAM3/results_v15}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets}"

# name:kf_switch:kf_min_gap
VARIANTS="
A0_orbslam3:0:0
G10:0:10
SW_G10:1:10
"

seq_dir()   { case "$1" in freiburg*) echo "$DATA_ROOT/rgbd_dataset_$1";; *) echo "$DATA_ROOT/rgbd_dataset_freiburg3_$1";; esac; }
seq_calib() { case "$1" in freiburg1*) echo Examples/RGB-D/TUM1.yaml;; freiburg2*) echo Examples/RGB-D/TUM2.yaml;; *) echo Examples/RGB-D/TUM3.yaml;; esac; }

for SEQ in $SEQS; do
  DATASET=$(seq_dir "$SEQ")
  CALIB=$(seq_calib "$SEQ")
  ASSOC=$DATASET/association_new.txt
  GT=$DATASET/groundtruth.txt
  if [ ! -d "$DATASET" ]; then echo "!! missing $DATASET - skipping $SEQ"; continue; fi
  if [ ! -s "$ASSOC" ]; then
    python3 evaluation/associate.py "$DATASET/rgb.txt" "$DATASET/depth.txt" > "$ASSOC"
    echo "created $ASSOC ($(wc -l < "$ASSOC") lines)"
  fi
  OUT=$ROOT/$SEQ
  mkdir -p "$OUT"
  echo "== $SEQ  calib=$CALIB"
  for VDEF in $VARIANTS; do
    IFS=: read -r V SW GAP <<< "$VDEF"
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" $V "* ]]; then continue; fi
    for i in $(seq 1 "$RUNS"); do
      echo "[$(date +%H:%M)] $SEQ  $V  run $i"
      DSLAM_MODE=baseline DSLAM_CHI2_GATE=raw DSLAM_LABELS=opt DSLAM_KERNEL=huber DSLAM_MAP_FILTER=0 \
      DSLAM_IGNORE= DSLAM_KF_SEMANTIC=0 DSLAM_KF_SWITCH=$SW DSLAM_KF_MIN_GAP=$GAP DSLAM_DIAG=0 \
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
    grep -h rmse $ROOT/$SEQ/${V}_*.eval | awk '{print $2}' | sort -g | awk -v s=$SEQ -v v=$V -v th=$FAIL_TH '
      {x[NR]=$1; t+=$1; if($1>th) f++}
      END {mu=t/NR; for(i=1;i<=NR;i++) d+=(x[i]-mu)^2; sd=(NR>1)?sqrt(d/(NR-1)):0;
           med=(NR%2)?x[(NR+1)/2]:(x[NR/2]+x[NR/2+1])/2;
           printf "%-22s %-12s n=%-2d mean=%.4f  median=%.4f  std=%.4f  fail=%d/%d\n", s, v, NR, mu, med, sd, f+0, NR}'
  done
done

echo; echo "===== KEYFRAMES (mean per run) ====="
for SEQ in $SEQS; do
  for F in $(ls $ROOT/$SEQ/*_1.eval 2>/dev/null); do
    V=$(basename "$F" _1.eval)
    for K in $ROOT/$SEQ/${V}_*_kf.txt; do [ -f "$K" ] && wc -l < "$K"; done | \
      awk -v s=$SEQ -v v=$V '{t+=$1; n++} END {if(n) printf "%-22s %-12s keyframes mean=%.1f (n=%d)\n", s, v, t/n, n}'
  done
done

echo; echo "===== YOLO (SW_G10 runs: share of frames where YOLO saw a person/vehicle) ====="
for SEQ in $SEQS; do
  grep -h "frames_with_detections" $ROOT/$SEQ/SW_G10_*.log 2>/dev/null | \
    sed 's/.*frames_with_detections=[0-9]* (\([0-9.]*\) %).*/\1/' | \
    awk -v s=$SEQ '{t+=$1; n++} END {if(n) printf "%-22s frames with detections: %.1f %% (mean of %d runs)\n", s, t/n, n}'
done
