#!/bin/bash
set -ex

AOM_VERSION="2.0.2-dmo0~bpo10+1"
PKG_URL="https://www.deb-multimedia.org/pool/main/a/aom-dmo"
ARCH="arm64"

cd "$DEPS_DIR"

[ -f "libaom-dev_${AOM_VERSION}_${ARCH}.deb" ] &&
[ -f "libaom2_${AOM_VERSION}_${ARCH}.deb" ] ||
curl -O "${PKG_URL}/libaom-dev_${AOM_VERSION}_${ARCH}.deb" \
     -O "${PKG_URL}/libaom2_${AOM_VERSION}_${ARCH}.deb"

sha256sum --check --ignore-missing <<EOF
2352aa82e15f3936c2dd21d3aee6633b8338e96c09b38b2912aa2c1555a758a2  libaom2_${AOM_VERSION}_${ARCH}.deb
80e7c9ea59f4fc9ac6518e071ee8f86ba2ccec2d5500ea222982fa6dfa21356c  libaom-dev_${AOM_VERSION}_${ARCH}.deb
EOF

sudo dpkg -i "libaom2_${AOM_VERSION}_${ARCH}.deb" \
             "libaom-dev_${AOM_VERSION}_${ARCH}.deb"
