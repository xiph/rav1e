#!/bin/bash
set -ex

DAV1D_VERSION="0.8.2-dmo1"
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
04d30fc34056467b91a627563c61b9a0046a2e084bb649791cd31887a6c76d8e  libdav1d-dev_${DAV1D_VERSION}_amd64.deb
0ec130514ce8748a84f4db3d624bf6f20e28dfb0f8a64659a75a8087642269fc  libdav1d-dev_${DAV1D_VERSION}_arm64.deb
0c3debb3a926e10009503e639dddcfd4082ed6e012340ca49682b738c243dedc  libdav1d5_${DAV1D_VERSION}_amd64.deb
3c29f1782d89f85ac1cc158560828d7e604c7070985e92b7c03135825af478cc  libdav1d5_${DAV1D_VERSION}_arm64.deb
EOF

sudo dpkg -i "libdav1d5_${DAV1D_VERSION}_$ARCH.deb" \
             "libdav1d-dev_${DAV1D_VERSION}_$ARCH.deb"
