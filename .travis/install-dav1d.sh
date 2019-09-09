#!/bin/bash
set -ex

DAV1D_VERSION="0.4.0"

# dav1d prints the version number to stderr
if [ "$(dav1d --version 2>&1 > /dev/null)" != "$DAV1D_VERSION" ]; then
  curl -L "https://code.videolan.org/videolan/dav1d/-/archive/$DAV1D_VERSION/dav1d-$DAV1D_VERSION.tar.gz" | tar xz
  cd "dav1d-$DAV1D_VERSION"
  # Tell meson where to look for nasm, because it doesn't respect our $PATH
  export NASM_PATH="$DEPS_DIR/bin/nasm"
  export NASM_PATH="${NASM_PATH//'/'/'\/'}"
  sed -i "s/nasm = find_program('nasm')/nasm = find_program(['nasm', '$NASM_PATH'])/g" meson.build
  meson build --buildtype release --prefix "$DEPS_DIR"
  ninja -C build install
else
  echo "Using cached directory."
fi
