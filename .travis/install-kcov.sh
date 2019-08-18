#!/bin/bash
set -ex

KCOV_VERSION="36"

if [ "$("$BUILD_DIR/kcov/bin/kcov" --version)" != "kcov $KCOV_VERSION" ]; then
  # Remove any old versions that might exist from the cache
  rm -rf "$BUILD_DIR/kcov"

  mkdir -p "$BUILD_DIR/kcov"
  curl -L "https://github.com/SimonKagstrom/kcov/archive/v$KCOV_VERSION.tar.gz" | tar xz
  cd "kcov-$KCOV_VERSION"
  mkdir .build && cd .build
  cmake -GNinja -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache .. -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/kcov" && ninja && ninja install
else
  echo "Using cached directory."
fi
