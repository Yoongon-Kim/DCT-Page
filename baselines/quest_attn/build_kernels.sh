#!/bin/bash
# Build Quest CUDA kernels for quest_attn baseline.
#
# Usage:
#   cd baselines/quest_attn
#   bash build_kernels.sh [/path/to/Quest/kernels]
#
# The Quest repo's kernels/ directory contains headers and 3rdparty deps
# (FlashInfer, pybind11) required for compilation. By default this script
# looks at $HOME/Quest/kernels; override with the first argument.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QUEST_KERNELS_DIR="${1:-$HOME/Quest/kernels}"

if [ ! -d "$QUEST_KERNELS_DIR" ]; then
    echo "ERROR: Quest kernels directory not found: $QUEST_KERNELS_DIR"
    echo "Usage: bash build_kernels.sh /path/to/Quest/kernels"
    exit 1
fi

QUEST_KERNELS_DIR="$(realpath "$QUEST_KERNELS_DIR")"
QUEST_REPO_DIR="$(realpath "$QUEST_KERNELS_DIR/..")"
echo "Using Quest kernels at: $QUEST_KERNELS_DIR"

# Ensure git submodules (flashinfer, pybind11) are initialized
if [ ! -f "$QUEST_KERNELS_DIR/3rdparty/flashinfer/CMakeLists.txt" ] || \
   [ ! -f "$QUEST_KERNELS_DIR/3rdparty/pybind/CMakeLists.txt" ]; then
    echo "Initializing Quest git submodules (flashinfer, pybind11)..."
    git -C "$QUEST_REPO_DIR" submodule update --init --recursive kernels/3rdparty/flashinfer kernels/3rdparty/pybind
fi

cd "$SCRIPT_DIR/ops"

# Clean previous build to avoid stale CMake cache issues
rm -rf build
mkdir -p build
cd build

cmake -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
      -DQUEST_KERNELS_DIR="$QUEST_KERNELS_DIR" \
      -GNinja ..
ninja

echo "Compilation finished."
cd ..

# Symlink the .so into quest_attn/ so Python can import quest_attn._kernels
for file in $(find "./build" -maxdepth 1 -name "*.so"); do
    abs_file=$(realpath "$file")
    if [ -e "$abs_file" ]; then
        ln -sf "$abs_file" "$SCRIPT_DIR/"
        echo "Linked $abs_file -> $SCRIPT_DIR/"
    fi
done
