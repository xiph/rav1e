#!/bin/bash
set -ex

SCCACHE_VERSION="0.2.10"

if [ "$(sccache --version)" != "sccache $SCCACHE_VERSION" ]; then
  curl -L "https://github.com/mozilla/sccache/releases/download/$SCCACHE_VERSION/sccache-$SCCACHE_VERSION-x86_64-unknown-linux-musl.tar.gz" | tar xz
  mv -f "sccache-$SCCACHE_VERSION-x86_64-unknown-linux-musl/sccache" "$DEPS_DIR/bin/sccache"
else
  echo "Using cached directory."
fi
