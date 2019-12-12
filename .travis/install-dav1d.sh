#!/bin/bash
set -ex

DAV1D_VERSION="0.5.2-dmo1"
PKG_URL="http://www.deb-multimedia.org/pool/main/d/dav1d-dmo"

case "$ARCH" in
  x86_64) ARCH=amd64 ;;
  aarch64) ARCH=arm64 ;;
esac

curl -O "$PKG_URL/libdav1d-dev_${DAV1D_VERSION}_$ARCH.deb" \
     -O "$PKG_URL/libdav1d3_${DAV1D_VERSION}_$ARCH.deb"

sha256sum --check --ignore-missing <<EOF
9d9618445d5f79867d7aa2e338e11b48386c961a3c58002523efe2bad3b3f815  libdav1d3_0.5.2-dmo1_amd64.deb
fb0e7d592fa2324c52ac3fb6f79d4ff1ec01923280afcac1bdd74beb174532ca  libdav1d3_0.5.2-dmo1_arm64.deb
9f61ffe439c95a934996a6c35ab986983bb36d93057f8d1c958fd839982b6cb1  libdav1d-dev_0.5.2-dmo1_amd64.deb
d2af1ebe3d953a07dc9512f45aac62be2622d361b3262faf71ac2ed9cb4bbfdd  libdav1d-dev_0.5.2-dmo1_arm64.deb
EOF

sudo dpkg -i "libdav1d3_${DAV1D_VERSION}_$ARCH.deb" \
             "libdav1d-dev_${DAV1D_VERSION}_$ARCH.deb"
