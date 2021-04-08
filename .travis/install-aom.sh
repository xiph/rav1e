#!/bin/bash
set -ex

AOM_VERSION="3.0.0-dmo0~bpo10+1"
PKG_URL="https://www.deb-multimedia.org/pool/main/a/aom-dmo"
ARCH="arm64"

cd "$DEPS_DIR"

[ -f "libaom-dev_${AOM_VERSION}_${ARCH}.deb" ] &&
[ -f "libaom2_${AOM_VERSION}_${ARCH}.deb" ] ||
curl -O "${PKG_URL}/libaom-dev_${AOM_VERSION}_${ARCH}.deb" \
     -O "${PKG_URL}/libaom3_${AOM_VERSION}_${ARCH}.deb"

sha256sum --check --ignore-missing <<EOF
1a9a7d34175871d96afa98981b1f4a60d84a420ee9cc3b1f61c7d2f1bdcb0ae5  libaom3_${AOM_VERSION}_${ARCH}.deb
aa0f6dd3ec62f2682d1d2322e305076e951b3118aee2d0adc5add7697ec88ff7  libaom-dev_${AOM_VERSION}_${ARCH}.deb
EOF

sudo dpkg -i "libaom3_${AOM_VERSION}_${ARCH}.deb" \
             "libaom-dev_${AOM_VERSION}_${ARCH}.deb"
