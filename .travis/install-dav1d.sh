#!/bin/bash
set -ex

DAV1D_VERSION="0.8.0-dmo1"
PKG_URL="https://www.deb-multimedia.org/pool/main/d/dav1d-dmo"

case "$ARCH" in
  x86_64) ARCH=amd64 ;;
  aarch64) ARCH=arm64 ;;
esac

cd "$DEPS_DIR"

[ -f "libdav1d-dev_${DAV1D_VERSION}_$ARCH.deb" ] &&
[ -f "libdav1d5_${DAV1D_VERSION}_$ARCH.deb" ] ||
curl -O "$PKG_URL/libdav1d-dev_${DAV1D_VERSION}_$ARCH.deb" \
     -O "$PKG_URL/libdav1d5_${DAV1D_VERSION}_$ARCH.deb"

sha256sum --check --ignore-missing <<EOF
207ff05de3caa20afb9f131fc369085d7f47204ab6b2903636145c154f965084  libdav1d-dev_${DAV1D_VERSION}_amd64.deb
e46ac8e8b69a47e7a93427eef0be91d654c130cb6ae3f3d74b85928d8ffa956c  libdav1d-dev_${DAV1D_VERSION}_arm64.deb
9c3f2c806ac3a1f3bbae55489aa421cbf381308320372e8c82f6aa225a82cb53  libdav1d5_${DAV1D_VERSION}_amd64.deb
70214eeab7690ac6dbb9eb65bd6faf9dafe821fa9e19d4b47888e1b002cf6b0d  libdav1d5_${DAV1D_VERSION}_arm64.deb
EOF

sudo dpkg -i "libdav1d5_${DAV1D_VERSION}_$ARCH.deb" \
             "libdav1d-dev_${DAV1D_VERSION}_$ARCH.deb"
