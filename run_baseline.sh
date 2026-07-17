#!/bin/bash

DATASET=~/datasets/rgbd_dataset_freiburg3_walking_xyz
ASSOC=$DATASET/association_new.txt
GT=$DATASET/groundtruth.txt

echo "====================================="
echo " RUN 1: BASELINE (set mbUseDynamic=false)"
echo "====================================="

rm -f baseline_*.txt

for i in 1
do
    echo "Baseline run $i"

    ./Examples/RGB-D/rgbd_tum \
    Vocabulary/ORBvoc.txt \
    Examples/RGB-D/TUM1.yaml \
    $DATASET \
    $ASSOC

    mv CameraTrajectory.txt baseline_$i.txt

    evo_ape tum $GT baseline_$i.txt -a > baseline_$i.eval
done

echo "Baseline RMSE:"
grep rmse baseline_*.eval


echo "Average RMSE:"
grep rmse baseline_*.eval | awk '{sum += $2} END {print sum/NR}'
