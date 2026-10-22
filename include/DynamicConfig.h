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
//   DSLAM_YOLO_LETTERBOX 1 = YOLOv5-style letterbox input (keep aspect ratio, pad with grey)
//                     0 = stretch the image to 640x640 (v3-v12 behaviour; on 640x480 TUM frames
//                         this lowers YOLOv5n confidence so much that nothing passes 0.4)
//                                                          (default: 1)
//   DSLAM_YOLO_GPU    1 = run YOLO on CUDA if OpenCV was built with CUDA (default: 1)
//                     0 = force CPU
//   DSLAM_CHI2_GATE   when and how PoseOptimization rejects outliers  (default: weighted)
//                     weighted = test the weighted chi2 after every round (v4/v5 behaviour;
//                                with w_geo the weighted chi2 never exceeds 7.36, so in
//                                practice nothing is rejected)
//                     raw      = test the unweighted chi2 after every round (ORB-SLAM3)
//                     late     = no rejection after rounds 0 and 1; unweighted test after
//                                rounds 2 and 3 (soft first, hard rejection only at the end)
//   DSLAM_LABELS      which outlier labels (mvbOutlier) PoseOptimization hands to the rest of
//                     ORB-SLAM3 (tracking discards, new keyframe points, map point statistics,
//                     keyframe decision). The optimisation itself is unchanged.   (default: opt)
//                     opt  = labels from the gate used during optimisation (normal behaviour)
//                     raw  = relabel after the last round with the unweighted chi2 test
//                     none = report every point as inlier
//   DSLAM_IGNORE      comma-separated list: places in Tracking that should ignore the outlier label
//                     (labels are still computed normally; only that one use is switched off)
//                     E1 = tracking discard: outlier points keep their map-point link
//                          (TrackReferenceKeyFrame / TrackWithMotionModel)
//                     E2 = pre-keyframe cleanup: outlier links are not removed before the next frame /
//                          CreateNewKeyFrame (which would otherwise create NEW map points there)
//                     E3 = map statistics: outliers also count as "found" (MapPoint::IncreaseFound)
//                     E4 = keyframe decision: outliers count as tracked close points (NeedNewKeyFrame)
//                     E5 = E5a + E5b
//                     E5a = outliers count in the inlier number for the tracking-OK test only
//                     E5b = outliers count in the inlier number for the keyframe decision only
//                           (mnMatchesInliersKF in NeedNewKeyFrame)
//                     E6 = frame-to-frame matching: last frame's outliers are still projected and
//                          matched into the current frame (ORBmatcher::SearchByProjection)
//                     e.g. DSLAM_IGNORE=E2   or   DSLAM_IGNORE=E1,E2,E3,E4,E5   (default: empty)
//   DSLAM_KF_SEMANTIC 1 = in E4 / E5b, ignore an outlier label ONLY if the keypoint lies inside a
//                         YOLO dynamic box (outliers on static scene parts still trigger keyframes).
//                         Loads YOLO (YOLO is then used for the keyframe decision only).
//                     0 = E4 / E5b ignore every outlier label           (default: 0)
//   DSLAM_KF_SWITCH   1 = the E4 / E5b rule is active only while YOLO sees dynamic objects:
//                         on frames where at least DSLAM_KF_SWITCH_MIN of the keypoints lie in YOLO
//                         boxes, and for DSLAM_KF_SWITCH_HOLD frames after the last such frame.
//                         Otherwise the keyframe decision is plain ORB-SLAM3.   (default: 0)
//   DSLAM_KF_SWITCH_MIN   fraction of keypoints inside boxes                (default: 0.05)
//   DSLAM_KF_SWITCH_HOLD  frames the rule stays on after the last detection (default: 30)
//   DSLAM_KF_MIN_GAP  N > 0 = at most one keyframe every N frames (plain rate cap)  (default: 0 = off)
//                     With DSLAM_KF_SWITCH=1 the cap applies only while YOLO sees dynamic objects.
//   DSLAM_DIAG        1 = write one CSV line per tracked frame to DSLAM_DIAG_FILE
//                         (default diag_frames.csv): outliers, outliers inside YOLO boxes, keyframe
//                         decision inputs, keyframe created, new map points inside/outside boxes.
//                         Loads YOLO (changes timing slightly; use for analysis, not for timing).
//   DSLAM_KERNEL      robust kernel in PoseOptimization                (default: huber)
//                     huber  = ORB-SLAM3 default (delta^2 = 5.991 mono / 7.815 stereo)
//                     cauchy = same deltas
//                     welsch = rho'(e2) = exp(-e2 / DSLAM_SIGMA): the same function as w_geo,
//                              but re-evaluated in every iteration instead of frozen after round 0
//   DSLAM_MAP_FILTER  1 = never create map points on keypoints inside a YOLO dynamic box
//                         (map initialisation, new keyframes, LocalMapping triangulation).
//                         Loads YOLO even in geo mode; YOLO is then used for the map only.
//                     0 = off                                 (default: 0)

#pragma once

#include <cstdlib>
#include <string>
#include <algorithm>
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
    bool        yoloGpu    = true;
    bool        yoloLetterbox = true; // DSLAM_YOLO_LETTERBOX
    enum Gate   { GATE_WEIGHTED = 0, GATE_RAW = 1, GATE_LATE = 2 };
    enum Kernel { KERNEL_HUBER = 0, KERNEL_CAUCHY = 1, KERNEL_WELSCH = 2 };
    enum Labels { LABELS_OPT = 0, LABELS_RAW = 1, LABELS_NONE = 2 };
    Gate        gate       = GATE_WEIGHTED;
    Labels      labels     = LABELS_OPT;
    bool        ignE1 = false, ignE2 = false, ignE3 = false, ignE4 = false, ignE5a = false, ignE5b = false, ignE6 = false;
    bool        kfSemantic = false;   // DSLAM_KF_SEMANTIC=1
    bool        kfSwitch   = false;   // DSLAM_KF_SWITCH=1
    float       kfSwitchMin  = 0.05f;
    int         kfSwitchHold = 30;
    int         kfSwitchPersist = 1;     // DSLAM_KF_SWITCH_PERSIST: YOLO must fire k frames in a row
    float       kfSwitchGeo     = 0.0f;  // DSLAM_KF_SWITCH_GEO: min outlier ratio inside boxes (0 = off)
    float       kfSwitchGeoRel  = 0.0f;  // DSLAM_KF_SWITCH_GEO_REL: and at least this much above the ratio outside
    int         kfSwitchGeoPts  = 10;    // DSLAM_KF_SWITCH_GEO_PTS: min matched map points inside boxes
    int         kfSwitchGeoMinConf = 1;  // DSLAM_KF_SWITCH_GEO_MINCONF: confirmations within the keep window
                                         // needed before YOLO alone may keep the switch on
    int         kfSwitchGeoKeep = 0;     // DSLAM_KF_SWITCH_GEO_KEEP: after a geometric confirmation, YOLO
                                         // alone keeps the switch on for this many frames (0 = off)
    int         kfMinGap   = 0;       // DSLAM_KF_MIN_GAP
    int         kfCapBypass = 0;      // DSLAM_KF_CAP_BYPASS: cap does not block a keyframe when fewer than
                                      // this many map points are tracked (tracking about to fail; 0 = off)
    bool        diag       = false;   // DSLAM_DIAG=1
    std::string diagFile   = "diag_frames.csv";
    Kernel      kernel     = KERNEL_HUBER;
    int         mapFilter  = 0;       // DSLAM_MAP_FILTER: 0 off, 1 no new map points inside YOLO boxes,
                                      // 2 the same but only while the keyframe switch is on (confirmed moving people)
    float       mapFilterDepth = 0.0f; // DSLAM_MAP_FILTER_DEPTH: keep only points within this depth [m] of the
                                       // person out of the map (0 = the whole box)

    bool UseYOLO()     const { return mode == YOLO || mode == PROPOSED; }
    bool UseGeometry() const { return mode == GEO  || mode == PROPOSED; }
    bool UseWeighting()const { return mode != BASELINE; }
    bool NeedYOLO()    const { return UseYOLO() || mapFilter || kfSemantic || kfSwitch || diag; }

    const char* GateName() const
    {
        return gate == GATE_RAW ? "raw" : gate == GATE_LATE ? "late" : "weighted";
    }
    const char* LabelsName() const
    {
        return labels == LABELS_RAW ? "raw" : labels == LABELS_NONE ? "none" : "opt";
    }
    const char* KernelName() const
    {
        return kernel == KERNEL_CAUCHY ? "cauchy" : kernel == KERNEL_WELSCH ? "welsch" : "huber";
    }

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
        if(const char* s = std::getenv("DSLAM_YOLO_GPU"))   c.yoloGpu   = std::atoi(s) != 0;
        if(const char* s = std::getenv("DSLAM_YOLO_LETTERBOX")) c.yoloLetterbox = std::atoi(s) != 0;
        if(const char* s = std::getenv("DSLAM_CHI2_GATE"))
        {
            std::string g(s);
            if(g == "raw")           c.gate = GATE_RAW;
            else if(g == "late")     c.gate = GATE_LATE;
            else if(g == "weighted") c.gate = GATE_WEIGHTED;
            else std::cerr << "[DynamicConfig] Unknown DSLAM_CHI2_GATE '" << g
                           << "', using 'weighted'" << std::endl;
        }
        if(const char* s = std::getenv("DSLAM_LABELS"))
        {
            std::string l(s);
            if(l == "raw")           c.labels = LABELS_RAW;
            else if(l == "none")     c.labels = LABELS_NONE;
            else if(l == "opt")      c.labels = LABELS_OPT;
            else std::cerr << "[DynamicConfig] Unknown DSLAM_LABELS '" << l
                           << "', using 'opt'" << std::endl;
        }
        if(const char* s = std::getenv("DSLAM_IGNORE"))
        {
            const std::string v = "," + std::string(s) + ",";
            c.ignE1 = v.find(",E1,") != std::string::npos;
            c.ignE2 = v.find(",E2,") != std::string::npos;
            c.ignE3 = v.find(",E3,") != std::string::npos;
            c.ignE4 = v.find(",E4,") != std::string::npos;
            const bool e5 = v.find(",E5,") != std::string::npos;
            c.ignE5a = e5 || v.find(",E5a,") != std::string::npos;
            c.ignE5b = e5 || v.find(",E5b,") != std::string::npos;
            c.ignE6 = v.find(",E6,") != std::string::npos;
        }
        if(const char* s = std::getenv("DSLAM_KF_SEMANTIC")) c.kfSemantic = std::atoi(s) != 0;
        if(const char* s = std::getenv("DSLAM_KF_SWITCH"))      c.kfSwitch     = std::atoi(s) != 0;
        if(const char* s = std::getenv("DSLAM_KF_SWITCH_MIN"))  c.kfSwitchMin  = std::stof(s);
        if(const char* s = std::getenv("DSLAM_KF_SWITCH_HOLD")) c.kfSwitchHold = std::atoi(s);
        if(const char* s = std::getenv("DSLAM_KF_SWITCH_PERSIST")) c.kfSwitchPersist = std::max(1, std::atoi(s));
        if(const char* s = std::getenv("DSLAM_KF_SWITCH_GEO"))     c.kfSwitchGeo     = std::stof(s);
        if(const char* s = std::getenv("DSLAM_KF_SWITCH_GEO_REL")) c.kfSwitchGeoRel  = std::stof(s);
        if(const char* s = std::getenv("DSLAM_KF_SWITCH_GEO_PTS")) c.kfSwitchGeoPts  = std::atoi(s);
        if(const char* s = std::getenv("DSLAM_KF_SWITCH_GEO_KEEP")) c.kfSwitchGeoKeep = std::atoi(s);
        if(const char* s = std::getenv("DSLAM_KF_SWITCH_GEO_MINCONF")) c.kfSwitchGeoMinConf = std::max(1, std::atoi(s));
        if(const char* s = std::getenv("DSLAM_KF_MIN_GAP"))     c.kfMinGap     = std::atoi(s);
        if(const char* s = std::getenv("DSLAM_KF_CAP_BYPASS"))  c.kfCapBypass  = std::atoi(s);
        if(const char* s = std::getenv("DSLAM_DIAG"))           c.diag         = std::atoi(s) != 0;
        if(const char* s = std::getenv("DSLAM_DIAG_FILE"))      c.diagFile     = s;
        if(const char* s = std::getenv("DSLAM_KERNEL"))
        {
            std::string k(s);
            if(k == "cauchy")        c.kernel = KERNEL_CAUCHY;
            else if(k == "welsch")   c.kernel = KERNEL_WELSCH;
            else if(k == "huber")    c.kernel = KERNEL_HUBER;
            else std::cerr << "[DynamicConfig] Unknown DSLAM_KERNEL '" << k
                           << "', using 'huber'" << std::endl;
        }
        if(const char* s = std::getenv("DSLAM_MAP_FILTER")) c.mapFilter = std::atoi(s);
        if(const char* s = std::getenv("DSLAM_MAP_FILTER_DEPTH")) c.mapFilterDepth = std::stof(s);

        std::cout << "[DynamicConfig] mode=" << c.ModeName()
                  << " sigma=" << c.sigma
                  << " min_sem=" << c.minSem
                  << " yolo_conf=" << c.yoloConf
                  << " second_opt=" << c.secondOpt
                  << " verbose=" << c.verbose
                  << " yolo_gpu=" << c.yoloGpu
                  << " yolo_letterbox=" << c.yoloLetterbox
                  << " chi2_gate=" << c.GateName()
                  << " kernel=" << c.KernelName()
                  << " labels=" << c.LabelsName()
                  << " ignore=" << (c.ignE1?"E1":"") << (c.ignE2?"E2":"") << (c.ignE3?"E3":"")
                  << (c.ignE4?"E4":"") << (c.ignE5a?"E5a":"") << (c.ignE5b?"E5b":"") << (c.ignE6?"E6":"")
                  << " kf_semantic=" << c.kfSemantic
                  << " kf_switch=" << c.kfSwitch << "(" << c.kfSwitchMin << "," << c.kfSwitchHold
                  << ",persist=" << c.kfSwitchPersist << ",geo=" << c.kfSwitchGeo << ",geo_rel=" << c.kfSwitchGeoRel
                  << ",geo_pts=" << c.kfSwitchGeoPts << ",geo_keep=" << c.kfSwitchGeoKeep << ",geo_minconf=" << c.kfSwitchGeoMinConf << ")"
                  << " kf_min_gap=" << c.kfMinGap << " kf_cap_bypass=" << c.kfCapBypass
                  << " diag=" << c.diag
                  << " map_filter=" << c.mapFilter << "(depth " << c.mapFilterDepth << ")" << std::endl;
        return c;
    }
};

} // namespace ORB_SLAM3
