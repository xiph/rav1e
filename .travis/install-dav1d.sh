#!/bin/bash
set -ex

DAV1D_VERSION="0.8.1-dmo1"
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
dcf911325699d93a90818e16736e2c93b29d8e7538c1545accd3b25c610876c0  libdav1d-dev_${DAV1D_VERSION}_amd64.deb
37094752ae6f8a4c1d6a8267b9632144a235a995f02c5f8bfc69cd8ffc0bb831  libdav1d-dev_${DAV1D_VERSION}_arm64.deb
06f51b9660d413417827270b298e2ad541bd8ddaae7e027ebcb6bb7b6b1ad006  libdav1d5_${DAV1D_VERSION}_amd64.deb
3f35ba159cb76108ba483aedae7acd6eb797bc7cf7a8b0023eeaede2f4b2fbb0  libdav1d5_${DAV1D_VERSION}_arm64.deb
EOF

sudo dpkg -i "libdav1d5_${DAV1D_VERSION}_$ARCH.deb" \
             "libdav1d-dev_${DAV1D_VERSION}_$ARCH.deb"
