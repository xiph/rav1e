#!/bin/bash
set -ex

NASM_VERSION="2.14"

if [[ "$(nasm --version)" != "NASM version $NASM_VERSION"* ]]; then
  curl -L "https://download.videolan.org/contrib/nasm/nasm-$NASM_VERSION.tar.gz" | tar xz
  cd "nasm-$NASM_VERSION"
  ./configure CC='sccache gcc' --prefix="$DEPS_DIR" && make -j2 && make install
else
  echo "Using cached directory."
fi
