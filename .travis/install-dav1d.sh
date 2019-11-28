#!/bin/bash
set -ex

DAV1D_VERSION="0.5.1-dmo1"
PKG_URL="http://www.deb-multimedia.org/pool/main/d/dav1d-dmo"

case "$ARCH" in
  x86_64) ARCH=amd64 ;;
  aarch64) ARCH=arm64 ;;
esac

curl -O "$PKG_URL/libdav1d-dev_${DAV1D_VERSION}_$ARCH.deb" \
     -O "$PKG_URL/libdav1d3_${DAV1D_VERSION}_$ARCH.deb"

sha256sum --check --ignore-missing <<EOF
682fd52fcfd73c225f9aaee200cbe69eceefdc687b8ce03f354731f5288a28ce  libdav1d3_0.5.1-dmo1_amd64.deb
de8550873cd7c7a7ede789f9be5db079cb3eafa91facc883dec833da887fa831  libdav1d3_0.5.1-dmo1_arm64.deb
feb8fd535ae7747963d3d17ed394dc5bdb3e6163fdb77df787ec72a8ca9aac2e  libdav1d-dev_0.5.1-dmo1_amd64.deb
a4ebf7794c9ac2b9eeef90386c4655c7f16e4e299435fccb41c815dd380d05da  libdav1d-dev_0.5.1-dmo1_arm64.deb
EOF

sudo dpkg -i "libdav1d3_${DAV1D_VERSION}_$ARCH.deb" \
             "libdav1d-dev_${DAV1D_VERSION}_$ARCH.deb"
