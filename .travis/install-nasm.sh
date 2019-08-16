#!/bin/sh
set -ex

NASM_VERSION="2.14"

if [ ! -d "$BUILD_DIR/nasm-$NASM_VERSION" ]; then
  # Remove any old versions that might exist from the cache
  rm -rf "$BUILD_DIR/nasm*"

  mkdir -p "$BUILD_DIR/nasm"
  curl -L "https://download.videolan.org/contrib/nasm/nasm-$NASM_VERSION.tar.gz" | tar xz
  cd "nasm-$NASM_VERSION"
  ./configure CC='sccache gcc' --prefix="$BUILD_DIR/nasm" && make -j2 && sudo make install
  cd ..
else
  echo "Using cached directory."
fi
