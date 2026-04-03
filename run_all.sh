#!/bin/bash

DATASET_PATH=~/datasets
VOC=Vocabulary/ORBvoc.txt
CONFIG=Examples/RGB-D/TUM1.yaml

SEQUENCES=(
freiburg3_walking_xyz
freiburg3_walking_static
freiburg3_walking_halfsphere
freiburg3_sitting_xyz
freiburg3_sitting_halfsphere
)

for SEQ in "${SEQUENCES[@]}"
do
    echo "Running $SEQ..."

    ./Examples/RGB-D/rgbd_tum \
    $VOC \
    $CONFIG \
    $DATASET_PATH/$SEQ \
    $DATASET_PATH/$SEQ/association.txt

    mv CameraTrajectory.txt traj_$SEQ.txt
done

echo "All sequences done."
