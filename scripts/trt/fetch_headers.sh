#!/usr/bin/env bash
# The pip TensorRT packages contain the libraries but not the C++ headers. This fetches the official headers for
# the installed version from github.com/NVIDIA/TensorRT (Apache-2.0) into Thirdparty/tensorrt/include,
# where CMakeLists.txt looks for them. Run from ~/ORB_SLAM3.
set -e
VER=$(python3 -c "import tensorrt; print(tensorrt.__version__)")
BRANCH="release/$(echo "$VER" | cut -d. -f1-2)"
echo "TensorRT $VER -> headers from branch $BRANCH"
DEST=Thirdparty/tensorrt
if [ -d "$DEST/.git" ]; then
  echo "$DEST already there; to refetch, move it away first (e.g. mv $DEST trash_old_runs/)"; exit 0
fi
git clone -q --depth 1 --branch "$BRANCH" --filter=blob:none --sparse https://github.com/NVIDIA/TensorRT.git "$DEST"
git -C "$DEST" sparse-checkout set include
ls "$DEST/include/NvInfer.h" && echo "headers ready in $DEST/include"
