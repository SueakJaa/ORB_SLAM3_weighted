#!/bin/bash
# v12: three follow-ups to the K_kf result (see doc "Dynamic SLAM Findings: v4 to v11").
#
#   A0_orbslam3   reference: plain ORB-SLAM3
#   K_kf          reference: outliers do not trigger keyframes (E4 + E5b)
#   SW_kf         K_kf rule ONLY while YOLO sees people (DSLAM_KF_SWITCH=1), plain ORB-SLAM3 otherwise
#                 -> goal: keep K_kf on dynamic scenes, remove the cost on the static office
#   G5/G10/G20    plain ORB-SLAM3 with at most one keyframe every 5/10/20 frames (no label change)
#                 -> is the whole effect just "fewer keyframes"?
#   DIAG_A0       1 run of A0 with per-frame CSV logging (DSLAM_DIAG=1)
#   DIAG_K_kf     1 run of K_kf with per-frame CSV logging
#                 -> where are the outliers? which keyframes add points on people?
#                    analyse with:  python3 scripts/analyze_diag.py results_v12/<seq>/DIAG_*_1_diag.csv
#
# Usage (from ~/ORB_SLAM3):
#   ./scripts/run_v12.sh                                     (4 sequences, 5 runs, ~2 h)
#   SEQS="walking_xyz" ONLY="SW_kf G10" ./scripts/run_v12.sh
# Runs with YOLO (SW_kf, DIAG_*) are not comparable for timing; use A0/K_kf/G* for runtime.

SEQS="${SEQS:-walking_xyz long_office_household walking_halfsphere walking_rpy}"
RUNS="${RUNS:-5}"
FAIL_TH="${FAIL_TH:-0.1}"
CALIB=Examples/RGB-D/TUM3.yaml
ROOT="${ROOT:-$HOME/ORB_SLAM3/results_v12}"

# name:ignore:kf_switch:kf_min_gap:diag:runs(empty = RUNS)
VARIANTS="
A0_orbslam3::0:0:0:
K_kf:E4,E5b:0:0:0:
SW_kf:E4,E5b:1:0:0:
G5::0:5:0:
G10::0:10:0:
G20::0:20:0:
DIAG_A0::0:0:1:1
DIAG_K_kf:E4,E5b:0:0:1:1
"

for SEQ in $SEQS; do
  DATASET=~/datasets/rgbd_dataset_freiburg3_$SEQ
  ASSOC=$DATASET/association_new.txt
  GT=$DATASET/groundtruth.txt
  OUT=$ROOT/$SEQ
  mkdir -p "$OUT"
  for VDEF in $VARIANTS; do
    IFS=: read -r V IGN SW GAP DIAG VR <<< "$VDEF"
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" $V "* ]]; then continue; fi
    NR="${VR:-$RUNS}"
    for i in $(seq 1 "$NR"); do
      echo "[$(date +%H:%M)] $SEQ  $V  run $i"
      rm -f diag_frames.csv
      DSLAM_MODE=baseline DSLAM_CHI2_GATE=raw DSLAM_LABELS=opt DSLAM_KERNEL=huber DSLAM_MAP_FILTER=0 \
      DSLAM_IGNORE=$IGN DSLAM_KF_SEMANTIC=0 DSLAM_KF_SWITCH=$SW DSLAM_KF_MIN_GAP=$GAP DSLAM_DIAG=$DIAG \
        ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt $CALIB $DATASET $ASSOC > "$OUT/${V}_$i.log" 2>&1
      if [ -f CameraTrajectory.txt ]; then
        mv CameraTrajectory.txt "$OUT/${V}_$i.txt"
        [ -f KeyFrameTrajectory.txt ] && mv KeyFrameTrajectory.txt "$OUT/${V}_${i}_kf.txt"
        [ -f diag_frames.csv ] && mv diag_frames.csv "$OUT/${V}_${i}_diag.csv"
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

echo; echo "===== TIME PER FRAME (mean tracking time; YOLO variants not comparable) ====="
for SEQ in $SEQS; do
  for F in $(ls $ROOT/$SEQ/*_1.eval 2>/dev/null); do
    V=$(basename "$F" _1.eval)
    grep -h "mean tracking time" $ROOT/$SEQ/${V}_*.log 2>/dev/null | \
      awk -v s=$SEQ -v v=$V '{t+=$4; n++} END {if(n) printf "%-15s %-16s %.1f ms\n", s, v, 1000*t/n}'
  done
done
