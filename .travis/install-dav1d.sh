#!/bin/bash
set -ex

DAV1D_VERSION="0.7.1-dmo0~bpo10+1"
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
2a1005c191f9ff41f53a5cea10b836e9e18f7f1390d81d935ef7de8a89223e02  libdav1d-dev_0.7.1-dmo0~bpo10+1_amd64.deb
199c222d620a40a4b6f3104c6fae351e7f7e96b4860432738cadf32a023ab91a  libdav1d-dev_0.7.1-dmo0~bpo10+1_arm64.deb
7274ea2516b32ca7714979a9a39073ec189dd9a874ccac70730fd1026bbc9b05  libdav1d4_0.7.1-dmo0~bpo10+1_amd64.deb
75ebfe5cce146c1a1b7aab40c9ee4ecb7f9423bc03ce074b8c2aa31ed59710bd  libdav1d4_0.7.1-dmo0~bpo10+1_arm64.deb
EOF

sudo dpkg -i "libdav1d4_${DAV1D_VERSION}_$ARCH.deb" \
             "libdav1d-dev_${DAV1D_VERSION}_$ARCH.deb"
