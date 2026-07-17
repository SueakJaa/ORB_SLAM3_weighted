#pragma once

namespace ORB_SLAM3
{

enum ExperimentMode
{
    BASELINE = 0,
    GEOMETRY_ONLY,
    YOLO_ONLY,
    YOLO_GEOMETRY,
    FULL_METHOD
};

// ======================================
// CHANGE ONLY THIS LINE FOR EXPERIMENTS
// ======================================

constexpr ExperimentMode EXPERIMENT = BASELINE;

}