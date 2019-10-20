#!/bin/bash
set -ex

SCCACHE_VERSION="0.2.10"

export RUSTC_WRAPPER=sccache

if [ "$(sccache --version)" = "sccache $SCCACHE_VERSION" ]; then
  echo "Using cached directory."
elif [ "$ARCH" = "x86_64" ]; then
  curl -L "https://github.com/mozilla/sccache/releases/download/$SCCACHE_VERSION/sccache-$SCCACHE_VERSION-x86_64-unknown-linux-musl.tar.gz" | tar xz
  mv -f "sccache-$SCCACHE_VERSION-x86_64-unknown-linux-musl/sccache" "$DEPS_DIR/bin/sccache"
else
  # cargo install --version "$SCCACHE_VERSION" --root "$DEPS_DIR" --no-default-features sccache
  echo "Skipping sccache installation on $ARCH."
  unset RUSTC_WRAPPER
fi

set +ex
