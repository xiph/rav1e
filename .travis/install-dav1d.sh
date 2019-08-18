#!/bin/bash
set -ex

DAV1D_VERSION="0.4.0"

# needed to check if dav1d exists and is the right version
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$BUILD_DIR/dav1d/lib/x86_64-linux-gnu"

# dav1d prints the version number to stderr
if [ "$("$BUILD_DIR/dav1d/bin/dav1d" --version 2>&1 > /dev/null)" != "$DAV1D_VERSION" ]; then
  # Remove any old versions that might exist from the cache
  rm -rf "$BUILD_DIR/dav1d"

  mkdir -p "$BUILD_DIR/dav1d"
  curl -L "https://code.videolan.org/videolan/dav1d/-/archive/$DAV1D_VERSION/dav1d-$DAV1D_VERSION.tar.gz" | tar xz
  cd "dav1d-$DAV1D_VERSION"
  # Tell meson where to look for nasm, because it doesn't respect our $PATH
  export NASM_PATH="$BUILD_DIR/nasm/bin/nasm"
  export NASM_PATH="${NASM_PATH//'/'/'\/'}"
  sed -i "s/nasm = find_program('nasm')/nasm = find_program(['nasm', '$NASM_PATH'])/g" meson.build
  meson build --buildtype release --prefix "$BUILD_DIR/dav1d"
  ninja -C build install
else
  echo "Using cached directory."
fi
