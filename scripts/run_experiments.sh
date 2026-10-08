#!/usr/bin/env bash
# Run all configurations x sequences x repeats and collect ATE RMSE.
#
# Usage:
#   ./scripts/run_experiments.sh                 # full run (4 modes x 3 seqs x 5 runs)
#   MODES="proposed" SEQS="walking_xyz" RUNS=1 ./scripts/run_experiments.sh   # quick test
#
# Edit the paths below to match your machine.

set -u

ORB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TUM_ROOT="${TUM_ROOT:-$HOME/datasets/TUM}"           # folder that contains rgbd_dataset_freiburg3_*
VOCAB="$ORB_ROOT/Vocabulary/ORBvoc.txt"
SETTINGS="$ORB_ROOT/Examples/RGB-D/TUM3.yaml"
EXE="$ORB_ROOT/Examples/RGB-D/rgbd_tum"

MODES="${MODES:-baseline geo yolo proposed}"
SEQS="${SEQS:-walking_xyz walking_static sitting_static}"
RUNS="${RUNS:-5}"
OUT_ROOT="${OUT_ROOT:-$ORB_ROOT/results/$(date +%Y%m%d_%H%M)}"

export DSLAM_VERBOSE=0          # no per-frame prints (needed for fair timing)
export DSLAM_SECOND_OPT="${DSLAM_SECOND_OPT:-0}"

mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/summary.csv"
echo "mode,sequence,run,ate_rmse_m,median_track_s,mean_track_s,mean_yolo_s" > "$SUMMARY"

for seq in $SEQS; do
    DATA="$TUM_ROOT/rgbd_dataset_freiburg3_${seq}"
    ASSOC="${ASSOC_NAME:+$DATA/$ASSOC_NAME}"; ASSOC="${ASSOC:-$DATA/associations.txt}"   # same file you used before
    GT="$DATA/groundtruth.txt"
    for mode in $MODES; do
        for run in $(seq 1 "$RUNS"); do
            RUN_DIR="$OUT_ROOT/$mode/$seq/run$run"
            mkdir -p "$RUN_DIR"
            echo ">>> mode=$mode seq=$seq run=$run"
            ( cd "$RUN_DIR" && DSLAM_MODE="$mode" "$EXE" "$VOCAB" "$SETTINGS" "$DATA" "$ASSOC" > log.txt 2>&1 )

            TRAJ="$RUN_DIR/CameraTrajectory.txt"
            if [ ! -s "$TRAJ" ]; then
                echo "$mode,$seq,$run,FAILED,,," >> "$SUMMARY"; continue
            fi

            # ATE RMSE, SE(3) alignment, no scale (RGB-D is metric).
            # IMPORTANT: use the same evaluation you used for the paper.
            ATE=$(evo_ape tum "$GT" "$TRAJ" -a 2>/dev/null | awk '/rmse/ {print $2}')

            MED=$(awk '/median tracking time/ {print $4}' "$RUN_DIR/log.txt")
            MEAN=$(awk '/mean tracking time/ {print $4}' "$RUN_DIR/log.txt")
            YOLO=$(grep -o 'mean_yolo_time=[0-9.e-]*' "$RUN_DIR/log.txt" | cut -d= -f2)
            echo "$mode,$seq,$run,$ATE,$MED,$MEAN,${YOLO:-0}" >> "$SUMMARY"
        done
    done
done

echo
echo "Done. Per-run results: $SUMMARY"
python3 - "$SUMMARY" <<'EOF'
import csv, sys, statistics as st
from collections import defaultdict
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["ate_rmse_m"] not in ("", "FAILED")]
g = defaultdict(list)
for r in rows: g[(r["mode"], r["sequence"])].append(float(r["ate_rmse_m"]))
print(f"{'mode':10s} {'sequence':16s} {'n':>2s} {'mean':>8s} {'std':>8s}")
for (m, s), v in sorted(g.items()):
    sd = st.stdev(v) if len(v) > 1 else 0.0
    print(f"{m:10s} {s:16s} {len(v):2d} {st.mean(v):8.4f} {sd:8.4f}")
EOF
