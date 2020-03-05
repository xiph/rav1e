#!/bin/bash
set -ex

DAV1D_VERSION="0.5.2-dmo2"
PKG_URL="http://www.deb-multimedia.org/pool/main/d/dav1d-dmo"

case "$ARCH" in
  x86_64) ARCH=amd64 ;;
  aarch64) ARCH=arm64 ;;
esac

cd "$DEPS_DIR"

[ -f "libdav1d-dev_${DAV1D_VERSION}_$ARCH.deb" ] &&
[ -f "libdav1d3_${DAV1D_VERSION}_$ARCH.deb" ] ||
curl -O "$PKG_URL/libdav1d-dev_${DAV1D_VERSION}_$ARCH.deb" \
     -O "$PKG_URL/libdav1d3_${DAV1D_VERSION}_$ARCH.deb"

sha256sum --check --ignore-missing <<EOF
918e83902927c9fbb17023a8973ecfea8876ac0deb2f5ffadf1d8cbbcbd4472f  libdav1d3_0.5.2-dmo2_amd64.deb
fd3e85c300b1b0f75b2092a0be694256e452e720b9d37e457a8cfb66cdbdbfb9  libdav1d3_0.5.2-dmo2_arm64.deb
c303be29c114d79b25f61b86b4e82fccbc748fbfb2484c5ba8f3936e5d8e90b7  libdav1d-dev_0.5.2-dmo2_amd64.deb
bb164a093e43172e48e47514c741974301e4120f98cf9a8411e21a47d4fecf9f  libdav1d-dev_0.5.2-dmo2_arm64.deb
EOF

sudo dpkg -i "libdav1d3_${DAV1D_VERSION}_$ARCH.deb" \
             "libdav1d-dev_${DAV1D_VERSION}_$ARCH.deb"
