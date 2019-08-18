#!/bin/bash
set -ex

AOM_VERSION="1.0.0-errata1"

if [[ "$("$BUILD_DIR/aom/bin/aomenc" --help)" != *"AV1 Encoder $AOM_VERSION"* ]]; then
  # Remove any old versions that might exist from the cache
  rm -rf "$BUILD_DIR/aom"

  mkdir -p "$BUILD_DIR/aom"
  git clone --depth 1 -b "v$AOM_VERSION" https://aomedia.googlesource.com/aom "aom-$AOM_VERSION"
  cd "aom-$AOM_VERSION"
  rm -rf CMakeCache.txt CMakeFiles
  mkdir -p .build
  cd .build
  cmake -GNinja .. -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=0 -DENABLE_DOCS=0 -DCONFIG_LOWBITDEPTH=1 -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/aom" -DCONFIG_PIC=1
  ninja && ninja install
else
  echo "Using cached directory."
fi
