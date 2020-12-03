#!/bin/bash
set -ex

AOM_VERSION="2.0.1-dmo0~bpo10+1"
PKG_URL="https://www.deb-multimedia.org/pool/main/a/aom-dmo"
ARCH="arm64"

cd "$DEPS_DIR"

[ -f "libaom-dev_${AOM_VERSION}_${ARCH}.deb" ] &&
[ -f "libaom2_${AOM_VERSION}_${ARCH}.deb" ] ||
curl -O "${PKG_URL}/libaom-dev_${AOM_VERSION}_${ARCH}.deb" \
     -O "${PKG_URL}/libaom2_${AOM_VERSION}_${ARCH}.deb"

sha256sum --check --ignore-missing <<EOF
26fcaf306ab6ca528fb26352460a00aa60b4f0d2cd1ba6c1de4af41352414c71  libaom2_${AOM_VERSION}_${ARCH}.deb
612b6f86a8dff9b6a4cd33216cdaf298605b6818159f8a9a056e7a73ce935481  libaom-dev_${AOM_VERSION}_${ARCH}.deb
EOF

sudo dpkg -i "libaom2_${AOM_VERSION}_${ARCH}.deb" \
             "libaom-dev_${AOM_VERSION}_${ARCH}.deb"
