#!/bin/bash
set -ex

DAV1D_VERSION="0.7.0-dmo1"
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
ade22c88d7a2307f4b6351f59bb6696504062ef5aaed88a2c0b6fe37085a20d1  libdav1d4_0.7.0-dmo1_amd64.deb
2db8f62c68f90bb0aafa2c6f183900d75d635ea9c99df15c8d9e5a606e036e74  libdav1d4_0.7.0-dmo1_arm64.deb
9ac5d588ad5db9cb6cd64eeb896305655f676838eef66115b82ab01272c3a504  libdav1d-dev_0.7.0-dmo1_amd64.deb
610ff6ec885a7f62f7d0256f640bb2a135c13a781b82f9aa267a0bd8a8749424  libdav1d-dev_0.7.0-dmo1_arm64.deb
EOF

sudo dpkg -i "libdav1d4_${DAV1D_VERSION}_$ARCH.deb" \
             "libdav1d-dev_${DAV1D_VERSION}_$ARCH.deb"
