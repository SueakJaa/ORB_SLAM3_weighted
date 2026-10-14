#!/bin/bash
# v16: make the YOLO switch robust to false detections.
#
# v15 showed that YOLO fires on 9-15 % of frames in static rooms (fr1_room, fr1_360), and each false
# "person" switches the keyframe cap on and costs accuracy. The new switch options:
#   DSLAM_KF_SWITCH_PERSIST=k   YOLO must see a person k frames in a row
#   DSLAM_KF_SWITCH_GEO=r       semantic + geometric: the matched map points INSIDE the boxes must be
#                               outliers of the pose optimisation (ratio >= r), i.e. really moving.
#                               A false detection on a chair has inliers inside its box.
#   DSLAM_KF_SWITCH_GEO_REL=d   ... and the ratio inside must be at least d above the ratio outside
#   DSLAM_KF_SWITCH_GEO_PTS=n   need at least n matched points inside the boxes (default 10)
#   DSLAM_KF_SWITCH_GEO_KEEP=N  (v17) hysteresis: geometry switches ON, then YOLO alone keeps it on
#                               for N frames after the last confirmation (variant GEOK_G10, GEO_KEEP)
#
# Two phases:
#   PHASE=diag  (default, ~20 min)  1 run per sequence with per-frame CSVs, for A0 and SW_G10.
#               Then:  python3 scripts/geo_switch_sim.py results_v16_diag
#               which simulates the switch options offline and suggests thresholds.
#   PHASE=eval  10 runs of A0, SW_G10, SW_P3 (persist 3), SW_C6 (YOLO conf 0.6), GEO_G10 (geometric)
#               GEO_R / GEO_REL set the geometric thresholds (defaults 0.5 / 0.2).
#
# Usage (from ~/ORB_SLAM3):
#   ./scripts/run_v16.sh 2>&1 | tee results_v16_diag_log.txt
#   PHASE=eval GEO_R=0.5 GEO_REL=0.2 ./scripts/run_v16.sh 2>&1 | tee results_v16_log.txt
#   PHASE=eval RUNS=5 SEQS="freiburg1_room walking_xyz" ONLY="SW_G10 GEO_G10" ./scripts/run_v16.sh

PHASE="${PHASE:-diag}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets}"
FAIL_TH="${FAIL_TH:-0.1}"
GEO_R="${GEO_R:-0.5}"
GEO_REL="${GEO_REL:-0.2}"
GEO_KEEP="${GEO_KEEP:-90}"
DEF_SEQS="walking_xyz walking_static walking_halfsphere sitting_static freiburg1_room freiburg1_desk freiburg1_desk2"

# name|extra environment (KF_SWITCH, KF_MIN_GAP, DIAG and switch options)
if [ "$PHASE" = "diag" ]; then
  SEQS="${SEQS:-$DEF_SEQS freiburg1_360}"
  RUNS="${RUNS:-1}"
  ROOT="${ROOT:-$HOME/ORB_SLAM3/results_v16_diag}"
  VARIANTS=(
    "DIAG_A0|DSLAM_KF_SWITCH=0 DSLAM_KF_MIN_GAP=0 DSLAM_DIAG=1"
    "DIAG_SW|DSLAM_KF_SWITCH=1 DSLAM_KF_MIN_GAP=10 DSLAM_DIAG=1"
  )
else
  SEQS="${SEQS:-$DEF_SEQS}"
  RUNS="${RUNS:-10}"
  ROOT="${ROOT:-$HOME/ORB_SLAM3/results_v16}"
  VARIANTS=(
    "A0_orbslam3|DSLAM_KF_SWITCH=0 DSLAM_KF_MIN_GAP=0"
    "SW_G10|DSLAM_KF_SWITCH=1 DSLAM_KF_MIN_GAP=10"
    "SW_P3|DSLAM_KF_SWITCH=1 DSLAM_KF_MIN_GAP=10 DSLAM_KF_SWITCH_PERSIST=3"
    "SW_C6|DSLAM_KF_SWITCH=1 DSLAM_KF_MIN_GAP=10 DSLAM_YOLO_CONF=0.6"
    "GEO_G10|DSLAM_KF_SWITCH=1 DSLAM_KF_MIN_GAP=10 DSLAM_KF_SWITCH_GEO=$GEO_R DSLAM_KF_SWITCH_GEO_REL=$GEO_REL"
    "GEOK_G10|DSLAM_KF_SWITCH=1 DSLAM_KF_MIN_GAP=10 DSLAM_KF_SWITCH_GEO=$GEO_R DSLAM_KF_SWITCH_GEO_REL=$GEO_REL DSLAM_KF_SWITCH_GEO_KEEP=$GEO_KEEP"
  )
fi

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
  for VDEF in "${VARIANTS[@]}"; do
    V="${VDEF%%|*}"; EXTRA="${VDEF#*|}"
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" $V "* ]]; then continue; fi
    for i in $(seq 1 "$RUNS"); do
      echo "[$(date +%H:%M)] $SEQ  $V  run $i"
      rm -f diag_frames.csv
      env DSLAM_MODE=baseline DSLAM_CHI2_GATE=raw DSLAM_LABELS=opt DSLAM_KERNEL=huber DSLAM_MAP_FILTER=0 \
          DSLAM_IGNORE= DSLAM_KF_SEMANTIC=0 DSLAM_DIAG=0 $EXTRA \
        ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt $CALIB $DATASET $ASSOC > "$OUT/${V}_$i.log" 2>&1
      [ -f diag_frames.csv ] && mv diag_frames.csv "$OUT/${V}_${i}_diag.csv"
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

echo; echo "===== KEYFRAMES (mean per run) and SWITCH (share of frames with the cap switched on) ====="
for SEQ in $SEQS; do
  for F in $(ls $ROOT/$SEQ/*_1.eval 2>/dev/null); do
    V=$(basename "$F" _1.eval)
    KF=$(for K in $ROOT/$SEQ/${V}_*_kf.txt; do [ -f "$K" ] && wc -l < "$K"; done | awk '{t+=$1; n++} END {if(n) printf "%.1f", t/n}')
    SWON=$(grep -h "switch_on_frames" $ROOT/$SEQ/${V}_*.log 2>/dev/null | \
      sed 's/.*switch_on_frames=[0-9]* (\([0-9.]*\) %).*/\1/' | awk '{t+=$1; n++} END {if(n) printf "%.1f %%", t/n; else printf "-"}')
    printf "%-22s %-12s keyframes=%-7s switch on=%s\n" "$SEQ" "$V" "$KF" "$SWON"
  done
done

if [ "$PHASE" = "diag" ]; then
  echo; echo "Next: python3 scripts/geo_switch_sim.py $ROOT"
fi
