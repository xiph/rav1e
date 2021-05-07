#!/bin/bash
set -ex

AOM_VERSION="3.1.0-dmo0~bpo10+1"
PKG_URL="https://www.deb-multimedia.org/pool/main/a/aom-dmo"
ARCH="arm64"

cd "$DEPS_DIR"

[ -f "libaom-dev_${AOM_VERSION}_${ARCH}.deb" ] &&
[ -f "libaom2_${AOM_VERSION}_${ARCH}.deb" ] ||
curl -O "${PKG_URL}/libaom-dev_${AOM_VERSION}_${ARCH}.deb" \
     -O "${PKG_URL}/libaom3_${AOM_VERSION}_${ARCH}.deb"

sha256sum --check --ignore-missing <<EOF
1846784bceba7d3c46c9672872c25292001aebd488a2035df49c9cae9a674c2a  libaom3_${AOM_VERSION}_${ARCH}.deb
1c2e509cd7fbb30304c2311f46509a68ebf939adaf9f7fd81cdc466866d06c05  libaom-dev_${AOM_VERSION}_${ARCH}.deb
EOF

sudo dpkg -i "libaom3_${AOM_VERSION}_${ARCH}.deb" \
             "libaom-dev_${AOM_VERSION}_${ARCH}.deb"
