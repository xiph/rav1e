#!/bin/bash
set -ex

KCOV_VERSION="36"

if [ "$(kcov --version)" != "kcov $KCOV_VERSION" ]; then
  curl -L "https://github.com/SimonKagstrom/kcov/archive/v$KCOV_VERSION.tar.gz" | tar xz
  cd "kcov-$KCOV_VERSION"
  mkdir .build && cd .build
  cmake -GNinja -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache .. -DCMAKE_INSTALL_PREFIX="$DEPS_DIR" && ninja && ninja install
else
  echo "Using cached directory."
fi
