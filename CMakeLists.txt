cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM3)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 4.4)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4.4 not found.")
   endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(realsense2)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# ---- optional TensorRT for the YOLO network (speed step 3, 26 Nov; scripts/trt/README.md) ----
# Found automatically: libraries from `pip install tensorrt...` (site-packages/tensorrt_libs) or a system/tar
# install; headers from scripts/trt/fetch_headers.sh (Thirdparty/tensorrt/include) or the system.
# Without TensorRT everything builds as before and YOLO uses OpenCV DNN.  -DWITH_TENSORRT=OFF to skip.
option(WITH_TENSORRT "Use TensorRT for YOLO if found" ON)
set(TRT_LIBS "")
if(WITH_TENSORRT)
  execute_process(COMMAND python3 -c "import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))"
                  OUTPUT_VARIABLE TRT_PIP_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            HINTS ${PROJECT_SOURCE_DIR}/Thirdparty/tensorrt/include $ENV{TENSORRT_ROOT}/include
                  /usr/include/x86_64-linux-gnu /usr/local/TensorRT/include)
  find_library(TENSORRT_LIB NAMES nvinfer libnvinfer.so.10 libnvinfer.so.8
               HINTS ${TRT_PIP_LIB_DIR} $ENV{TENSORRT_ROOT}/lib /usr/lib/x86_64-linux-gnu /usr/local/TensorRT/lib)
  find_path(CUDART_INCLUDE_DIR cuda_runtime_api.h HINTS /usr/local/cuda/include)
  find_library(CUDART_LIB cudart HINTS /usr/local/cuda/lib64)
  if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIB AND CUDART_INCLUDE_DIR AND CUDART_LIB)
    add_definitions(-DUSE_TENSORRT)
    include_directories(${TENSORRT_INCLUDE_DIR} ${CUDART_INCLUDE_DIR})
    set(TRT_LIBS ${TENSORRT_LIB} ${CUDART_LIB})
    message(STATUS "TensorRT: ${TENSORRT_LIB}  headers: ${TENSORRT_INCLUDE_DIR}")
  else()
    message(STATUS "TensorRT not found (lib '${TENSORRT_LIB}', headers '${TENSORRT_INCLUDE_DIR}'): YOLO uses OpenCV DNN")
  endif()
endif()

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Atlas.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Viewer.cc
src/ImuTypes.cc
src/G2oTypes.cc
src/CameraModels/Pinhole.cpp
src/CameraModels/KannalaBrandt8.cpp
src/OptimizableTypes.cpp
src/MLPnPsolver.cpp
src/GeometricTools.cc
src/TwoViewReconstruction.cc
src/Config.cc
src/Settings.cc
src/YOLOdetector.cc
src/TrtEngine.cc
include/System.h
include/Tracking.h
include/LocalMapping.h
include/LoopClosing.h
include/ORBextractor.h
include/ORBmatcher.h
include/FrameDrawer.h
include/Converter.h
include/MapPoint.h
include/KeyFrame.h
include/Atlas.h
include/Map.h
include/MapDrawer.h
include/Optimizer.h
include/Frame.h
include/KeyFrameDatabase.h
include/Sim3Solver.h
include/Viewer.h
include/ImuTypes.h
include/G2oTypes.h
include/CameraModels/GeometricCamera.h
include/CameraModels/Pinhole.h
include/CameraModels/KannalaBrandt8.h
include/OptimizableTypes.h
include/MLPnPsolver.h
include/GeometricTools.h
include/TwoViewReconstruction.h
include/SerializationUtils.h
include/Config.h
include/Settings.h)

add_subdirectory(Thirdparty/g2o)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
-lboost_serialization
-lcrypto
${TRT_LIBS}
)

# If RealSense SDK is found the library is added and its examples compiled
if(realsense2_FOUND)
    include_directories(${PROJECT_NAME}
    ${realsense_INCLUDE_DIR}
    )
    target_link_libraries(${PROJECT_NAME}
    ${realsense2_LIBRARY}
    )
endif()


# Build examples

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
        Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

# speed step 2: YOLO network benchmark (CPU / CUDA / CUDA FP16), binary in tools/
add_executable(yolo_bench tools/yolo_bench.cc src/TrtEngine.cc)
target_link_libraries(yolo_bench ${OpenCV_LIBS} ${TRT_LIBS})
set_target_properties(yolo_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/tools)

if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i
            Examples/RGB-D/rgbd_realsense_D435i.cc)
    target_link_libraries(rgbd_realsense_D435i ${PROJECT_NAME})
endif()


# RGB-D inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D-Inertial)

if(realsense2_FOUND)
    add_executable(rgbd_inertial_realsense_D435i
            Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
    target_link_libraries(rgbd_inertial_realsense_D435i ${PROJECT_NAME})
endif()

#Stereo examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

add_executable(stereo_kitti
        Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc
        Examples/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})

add_executable(stereo_tum_vi
        Examples/Stereo/stereo_tum_vi.cc)
target_link_libraries(stereo_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_realsense_t265
            Examples/Stereo/stereo_realsense_t265.cc)
    target_link_libraries(stereo_realsense_t265 ${PROJECT_NAME})

    add_executable(stereo_realsense_D435i
            Examples/Stereo/stereo_realsense_D435i.cc)
    target_link_libraries(stereo_realsense_D435i ${PROJECT_NAME})
endif()

#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_tum
        Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
        Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc
        Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})

add_executable(mono_tum_vi
        Examples/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_realsense_t265
            Examples/Monocular/mono_realsense_t265.cc)
    target_link_libraries(mono_realsense_t265 ${PROJECT_NAME})

    add_executable(mono_realsense_D435i
            Examples/Monocular/mono_realsense_D435i.cc)
    target_link_libraries(mono_realsense_D435i ${PROJECT_NAME})
endif()

#Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Inertial)

add_executable(mono_inertial_euroc
        Examples/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc ${PROJECT_NAME})

add_executable(mono_inertial_tum_vi
        Examples/Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_inertial_realsense_t265
            Examples/Monocular-Inertial/mono_inertial_realsense_t265.cc)
    target_link_libraries(mono_inertial_realsense_t265 ${PROJECT_NAME})

    add_executable(mono_inertial_realsense_D435i
            Examples/Monocular-Inertial/mono_inertial_realsense_D435i.cc)
    target_link_libraries(mono_inertial_realsense_D435i ${PROJECT_NAME})
endif()

#Stereo Inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo-Inertial)

add_executable(stereo_inertial_euroc
        Examples/Stereo-Inertial/stereo_inertial_euroc.cc)
target_link_libraries(stereo_inertial_euroc ${PROJECT_NAME})

add_executable(stereo_inertial_tum_vi
        Examples/Stereo-Inertial/stereo_inertial_tum_vi.cc)
target_link_libraries(stereo_inertial_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_inertial_realsense_t265
            Examples/Stereo-Inertial/stereo_inertial_realsense_t265.cc)
    target_link_libraries(stereo_inertial_realsense_t265 ${PROJECT_NAME})

    add_executable(stereo_inertial_realsense_D435i
            Examples/Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
    target_link_libraries(stereo_inertial_realsense_D435i ${PROJECT_NAME})
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Calibration)
if(realsense2_FOUND)
    add_executable(recorder_realsense_D435i
            Examples/Calibration/recorder_realsense_D435i.cc)
    target_link_libraries(recorder_realsense_D435i ${PROJECT_NAME})

    add_executable(recorder_realsense_T265
            Examples/Calibration/recorder_realsense_T265.cc)
    target_link_libraries(recorder_realsense_T265 ${PROJECT_NAME})
endif()

#Old examples

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/RGB-D)

add_executable(rgbd_tum_old
        Examples_old/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i_old
            Examples_old/RGB-D/rgbd_realsense_D435i.cc)
    target_link_libraries(rgbd_realsense_D435i_old ${PROJECT_NAME})
endif()


# RGB-D inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/RGB-D-Inertial)

if(realsense2_FOUND)
    add_executable(rgbd_inertial_realsense_D435i_old
            Examples_old/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
    target_link_libraries(rgbd_inertial_realsense_D435i_old ${PROJECT_NAME})
endif()

#Stereo examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Stereo)

add_executable(stereo_kitti_old
        Examples_old/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti_old ${PROJECT_NAME})

add_executable(stereo_euroc_old
        Examples_old/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc_old ${PROJECT_NAME})

add_executable(stereo_tum_vi_old
        Examples_old/Stereo/stereo_tum_vi.cc)
target_link_libraries(stereo_tum_vi_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_realsense_t265_old
            Examples_old/Stereo/stereo_realsense_t265.cc)
    target_link_libraries(stereo_realsense_t265_old ${PROJECT_NAME})

    add_executable(stereo_realsense_D435i_old
            Examples_old/Stereo/stereo_realsense_D435i.cc)
    target_link_libraries(stereo_realsense_D435i_old ${PROJECT_NAME})
endif()

#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Monocular)

add_executable(mono_tum_old
        Examples_old/Monocular/mono_tum.cc)
target_link_libraries(mono_tum_old ${PROJECT_NAME})

add_executable(mono_kitti_old
        Examples_old/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti_old ${PROJECT_NAME})

add_executable(mono_euroc_old
        Examples_old/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc_old ${PROJECT_NAME})

add_executable(mono_tum_vi_old
        Examples_old/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_realsense_t265_old
            Examples_old/Monocular/mono_realsense_t265.cc)
    target_link_libraries(mono_realsense_t265_old ${PROJECT_NAME})

    add_executable(mono_realsense_D435i_old
            Examples_old/Monocular/mono_realsense_D435i.cc)
    target_link_libraries(mono_realsense_D435i_old ${PROJECT_NAME})
endif()

#Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Monocular-Inertial)

add_executable(mono_inertial_euroc_old
        Examples_old/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc_old ${PROJECT_NAME})

add_executable(mono_inertial_tum_vi_old
        Examples_old/Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_inertial_realsense_t265_old
            Examples_old/Monocular-Inertial/mono_inertial_realsense_t265.cc)
    target_link_libraries(mono_inertial_realsense_t265_old ${PROJECT_NAME})

    add_executable(mono_inertial_realsense_D435i_old
            Examples_old/Monocular-Inertial/mono_inertial_realsense_D435i.cc)
    target_link_libraries(mono_inertial_realsense_D435i_old ${PROJECT_NAME})
endif()

#Stereo Inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Stereo-Inertial)

add_executable(stereo_inertial_euroc_old
        Examples_old/Stereo-Inertial/stereo_inertial_euroc.cc)
target_link_libraries(stereo_inertial_euroc_old ${PROJECT_NAME})

add_executable(stereo_inertial_tum_vi_old
        Examples_old/Stereo-Inertial/stereo_inertial_tum_vi.cc)
target_link_libraries(stereo_inertial_tum_vi_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_inertial_realsense_t265_old
            Examples_old/Stereo-Inertial/stereo_inertial_realsense_t265.cc)
    target_link_libraries(stereo_inertial_realsense_t265_old ${PROJECT_NAME})

    add_executable(stereo_inertial_realsense_D435i_old
            Examples_old/Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
    target_link_libraries(stereo_inertial_realsense_D435i_old ${PROJECT_NAME})
endif()
