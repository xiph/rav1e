#!/bin/bash
set -ex

DAV1D_VERSION="0.9.0-dmo1"
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
ce6bd5c710d287306d3b6d45fa3843b35231da37f4d18d82ff24ba088916cfae  libdav1d-dev_${DAV1D_VERSION}_amd64.deb
f415b4453a044d311426658b36b73efc0e13dcf9876923d1b1c661fb51e5d5b1  libdav1d-dev_${DAV1D_VERSION}_arm64.deb
54c8ff504523101b96fa994963fb24b7104221a5b011f8b525baac8260640994  libdav1d5_${DAV1D_VERSION}_amd64.deb
ee8af3bb6d2204477291f007b484394d9c100f6f55e11be22af3605b9f83282b  libdav1d5_${DAV1D_VERSION}_arm64.deb
EOF

sudo dpkg -i "libdav1d5_${DAV1D_VERSION}_$ARCH.deb" \
             "libdav1d-dev_${DAV1D_VERSION}_$ARCH.deb"
