#!/bin/bash
set -ex

DAV1D_VERSION="0.6.0-dmo1"
PKG_URL="http://www.deb-multimedia.org/pool/main/d/dav1d-dmo"

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
9a2a2bfd85b0ed814f91685b14ab78ae0c2b228c4e0b6b2f3f35e8d368713cdd  libdav1d4_0.6.0-dmo1_amd64.deb
2f276056c136d859a03c35bea60d1bc91147c1f6f7d769a705675ab1d7474112  libdav1d4_0.6.0-dmo1_arm64.deb
089dd451183e5b545882209794a68674db589f9880e3e5cf30f878d21bfb0a08  libdav1d-dev_0.6.0-dmo1_amd64.deb
b9dd34ba4d160bd3ea288391bc092cb857dbb64d0e31263efefdee7412f35228  libdav1d-dev_0.6.0-dmo1_arm64.deb
EOF

sudo dpkg -i "libdav1d4_${DAV1D_VERSION}_$ARCH.deb" \
             "libdav1d-dev_${DAV1D_VERSION}_$ARCH.deb"
