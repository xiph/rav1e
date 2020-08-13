#!/bin/bash
set -ex

AOM_VERSION="2.0.0-dmo0~bpo10+1"
PKG_URL="https://www.deb-multimedia.org/pool/main/a/aom-dmo"
ARCH="arm64"

cd "$DEPS_DIR"

[ -f "libaom-dev_${AOM_VERSION}_${ARCH}.deb" ] &&
[ -f "libaom2_${AOM_VERSION}_${ARCH}.deb" ] ||
curl -O "${PKG_URL}/libaom-dev_${AOM_VERSION}_${ARCH}.deb" \
     -O "${PKG_URL}/libaom2_${AOM_VERSION}_${ARCH}.deb"

sha256sum --check --ignore-missing <<EOF
dc485d96d9d9154469d4768f6f2da477c10293028a96a8412425f07bbc189be6  libaom2_${AOM_VERSION}_${ARCH}.deb
ee7fc5655d936050330a7516b44d872ef02eb1028a4ef6e801791f00f9a4caed  libaom-dev_${AOM_VERSION}_${ARCH}.deb
EOF

sudo dpkg -i "libaom2_${AOM_VERSION}_${ARCH}.deb" \
             "libaom-dev_${AOM_VERSION}_${ARCH}.deb"
