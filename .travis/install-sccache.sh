#!/bin/sh
set -ex

SCCACHE_VERSION="0.2.10"

if [ ! -d "$BUILD_DIR/sccache-$SCCACHE_VERSION-x86_64-unknown-linux-musl" ]; then
  # Remove any old versions that might exist from the cache
  rm -rf "$BUILD_DIR/sccache*"
  mkdir -p "$BUILD_DIR/sccache"

  curl -L "https://github.com/mozilla/sccache/releases/download/$SCCACHE_VERSION/sccache-$SCCACHE_VERSION-x86_64-unknown-linux-musl.tar.gz" | tar xz
  mv "sccache-$SCCACHE_VERSION-x86_64-unknown-linux-musl/sccache" "$BUILD_DIR/sccache/sccache"
else
  echo "Using cached directory."
fi
