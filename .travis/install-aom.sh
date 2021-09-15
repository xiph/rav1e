#!/bin/bash
set -ex

AOM_VERSION="3.1.1-dmo0~bpo10+1"
LIBVMAF_VERSION="2.1.1-dmo0~bpo10+3"
PKG_URL="https://www.deb-multimedia.org/pool/main/a/aom-dmo"
LIBVMAF_URL="https://www.deb-multimedia.org/pool/main/v/vmaf-dmo"
ARCH="arm64"

cd "$DEPS_DIR"

[ -f "libvmaf-dev_${LIBVMAF_VERSION}_${ARCH}.deb" ] && [ -f "libvmaf1_${LIBVMAF_VERSION}_${ARCH}.deb" ] && [ -f "libaom-dev_${AOM_VERSION}_${ARCH}.deb" ] &&
[ -f "libaom2_${AOM_VERSION}_${ARCH}.deb" ] ||
curl -O "${LIBVMAF_URL}/libvmaf1_${LIBVMAF_VERSION}_${ARCH}.deb" \
     -O "${LIBVMAF_URL}/libvmaf-dev_${LIBVMAF_VERSION}_${ARCH}.deb" \
     -O "${PKG_URL}/libaom-dev_${AOM_VERSION}_${ARCH}.deb" \
     -O "${PKG_URL}/libaom3_${AOM_VERSION}_${ARCH}.deb"

sha256sum --check --ignore-missing <<EOF
fe9321dd8d5901ddf74e407c1b213243c357430a03fad17249ec4d07c3cf8e93  libaom3_${AOM_VERSION}_${ARCH}.deb
f926f0af6db4faac5f9bd67051115ce3dfa4324ddb41647e773d17ff11fa8f3   libaom-dev_${AOM_VERSION}_${ARCH}.deb
d9dd550ab3a296019333ced63b80d47743eacef072b17cad96a990f67f587a42  libvmaf1_${LIBVMAF_VERSION}_${ARCH}.deb
ea706661c22df60005200608f54e29fc7f1cf41b47c1fc2def9df56dea10eac1  libvmaf-dev_${LIBVMAF_VERSION}_${ARCH}.deb
EOF

sudo dpkg -i "libvmaf1_${LIBVMAF_VERSION}_${ARCH}.deb" \
	     "libvmaf-dev_${LIBVMAF_VERSION}_${ARCH}.deb" \
	     "libaom3_${AOM_VERSION}_${ARCH}.deb" \
             "libaom-dev_${AOM_VERSION}_${ARCH}.deb"
