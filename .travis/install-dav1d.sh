#!/bin/bash
set -ex

DAV1D_VERSION="0.7.1-dmo1"
PKG_URL="https://www.deb-multimedia.org/pool/main/d/dav1d-dmo"

case "$ARCH" in
  x86_64) ARCH=amd64 ;;
  aarch64) ARCH=arm64 ;;
esac

cd "$DEPS_DIR"

[ -f "libdav1d-dev_${DAV1D_VERSION}_$ARCH.deb" ] &&
[ -f "libdav1d4_${DAV1D_VERSION}_$ARCH.deb" ] ||
curl -O "$PKG_URL/libdav1d-dev_${DAV1D_VERSION}_$ARCH.deb" \
     -O "$PKG_URL/libdav1d4_${DAV1D_VERSION}_$ARCH.deb"

sha256sum --check --ignore-missing <<EOF
6be3f602340dfcac1ce637dfd10cc7ab181e6b0d0089d934f9ebffffedc5d614  libdav1d-dev_${DAV1D_VERSION}_amd64.deb
e3c89addfc9df116558b1862954daea2ff9e1b621da9af76532dc1f72e5ec427  libdav1d-dev_${DAV1D_VERSION}_arm64.deb
47c8dbca45a5255799628ed994a7f8538fb10d18d231db5c4b8f75422f17e440  libdav1d4_${DAV1D_VERSION}_amd64.deb
09c4313a6f104af29d6b2aa66c64de494b96006df2216cd0196e91a590218856  libdav1d4_${DAV1D_VERSION}_arm64.deb
EOF

sudo dpkg -i "libdav1d4_${DAV1D_VERSION}_$ARCH.deb" \
             "libdav1d-dev_${DAV1D_VERSION}_$ARCH.deb"
