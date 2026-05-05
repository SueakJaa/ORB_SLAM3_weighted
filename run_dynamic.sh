#!/bin/bash

DATASET=~/datasets/rgbd_dataset_freiburg3_sitting_static
ASSOC=$DATASET/association_new.txt
GT=$DATASET/groundtruth.txt

echo "====================================="
echo " RUN: DYNAMIC (set mbUseDynamic=true)"
echo "====================================="

rm -f dynamic_*.txt dynamic_*.eval

for i in 1 2 3 4 5
do
    echo "Dynamic run $i"

    ./Examples/RGB-D/rgbd_tum \
    Vocabulary/ORBvoc.txt \
    Examples/RGB-D/TUM1.yaml \
    $DATASET \
    $ASSOC

    mv CameraTrajectory.txt dynamic_$i.txt

    evo_ape tum $GT dynamic_$i.txt -a > dynamic_$i.eval
done

echo "Dynamic RMSE:"
grep rmse dynamic_*.eval

echo "Average RMSE:"
grep rmse dynamic_*.eval | awk '{sum += $2} END {print sum/NR}'
