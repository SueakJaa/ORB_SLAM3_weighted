// DynamicConfig.h
// Run-time switches for the semantic-geometric weighting experiments.
// Every value is read once from an environment variable, so all four
// configurations run from the same binary (no more commenting lines in/out).
//
//   DSLAM_MODE        baseline | geo | yolo | proposed   (default: proposed)
//                       baseline : plain ORB-SLAM3 (YOLO not loaded, no weighting)
//                       geo      : w = exp(-chi2 / sigma)
//                       yolo     : w = max(min_sem, 1 - c)
//                       proposed : w = w_sem * w_geo
//   DSLAM_SIGMA       geometric decay sigma                (default: 20)
//   DSLAM_MIN_SEM     minimum semantic weight              (default: 0.3)
//   DSLAM_YOLO_CONF   YOLO confidence threshold            (default: 0.4)
//   DSLAM_YOLO_MODEL  path to yolov5n.onnx                 (default: old hard-coded path)
//   DSLAM_SECOND_OPT  1 = extra PoseOptimization after TrackLocalMap (old v3 behaviour)
//                     0 = paper behaviour                  (default: 0)
//   DSLAM_VERBOSE     1 = per-frame debug prints           (default: 0, keep 0 for timing)

#pragma once

#include <cstdlib>
#include <string>
#include <iostream>

namespace ORB_SLAM3
{

struct DynamicConfig
{
    enum Mode { BASELINE = 0, GEO = 1, YOLO = 2, PROPOSED = 3 };

    Mode        mode       = PROPOSED;
    float       sigma      = 20.0f;
    float       minSem     = 0.3f;
    float       yoloConf   = 0.4f;
    std::string yoloModel  = "/home/shirt/yolov5_models/yolov5n.onnx";
    bool        secondOpt  = false;
    bool        verbose    = false;

    bool UseYOLO()     const { return mode == YOLO || mode == PROPOSED; }
    bool UseGeometry() const { return mode == GEO  || mode == PROPOSED; }
    bool UseWeighting()const { return mode != BASELINE; }

    const char* ModeName() const
    {
        switch(mode)
        {
            case BASELINE: return "baseline";
            case GEO:      return "geo";
            case YOLO:     return "yolo";
            default:       return "proposed";
        }
    }

    static const DynamicConfig& Get()
    {
        static DynamicConfig cfg = Load();
        return cfg;
    }

private:
    static DynamicConfig Load()
    {
        DynamicConfig c;
        if(const char* s = std::getenv("DSLAM_MODE"))
        {
            std::string m(s);
            if(m == "baseline")      c.mode = BASELINE;
            else if(m == "geo")      c.mode = GEO;
            else if(m == "yolo")     c.mode = YOLO;
            else if(m == "proposed") c.mode = PROPOSED;
            else std::cerr << "[DynamicConfig] Unknown DSLAM_MODE '" << m
                           << "', using 'proposed'" << std::endl;
        }
        if(const char* s = std::getenv("DSLAM_SIGMA"))      c.sigma     = std::stof(s);
        if(const char* s = std::getenv("DSLAM_MIN_SEM"))    c.minSem    = std::stof(s);
        if(const char* s = std::getenv("DSLAM_YOLO_CONF"))  c.yoloConf  = std::stof(s);
        if(const char* s = std::getenv("DSLAM_YOLO_MODEL")) c.yoloModel = s;
        if(const char* s = std::getenv("DSLAM_SECOND_OPT")) c.secondOpt = std::atoi(s) != 0;
        if(const char* s = std::getenv("DSLAM_VERBOSE"))    c.verbose   = std::atoi(s) != 0;

        std::cout << "[DynamicConfig] mode=" << c.ModeName()
                  << " sigma=" << c.sigma
                  << " min_sem=" << c.minSem
                  << " yolo_conf=" << c.yoloConf
                  << " second_opt=" << c.secondOpt
                  << " verbose=" << c.verbose << std::endl;
        return c;
    }
};

} // namespace ORB_SLAM3
