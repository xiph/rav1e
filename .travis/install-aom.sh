#!/bin/bash
set -ex

AOM_VERSION="1.0.0.errata1-3"
PKG_URL="http://http.us.debian.org/debian/pool/main/a/aom"

case "$ARCH" in
  x86_64) ARCH=amd64 ;;
  aarch64) ARCH=arm64 ;;
  ppc64el) ARCH=ppc64le ;;
  s390x) ARCH=s390x ;;
esac

cd "$DEPS_DIR"

[ -f "libaom-dev_${AOM_VERSION}_$ARCH.deb" ] &&
[ -f "libaom0_${AOM_VERSION}_$ARCH.deb" ] ||
curl -O "$PKG_URL/libaom-dev_${AOM_VERSION}_$ARCH.deb" \
     -O "$PKG_URL/libaom0_${AOM_VERSION}_$ARCH.deb"

sha256sum --check --ignore-missing <<EOF
900f94cd878e6ba2acf87a2a324838736d5085b436f9bf615b2a3ed0345f8a0d  libaom0_1.0.0.errata1-3_amd64.deb
600536f50bf36cbcfabfc8eacb43a8f26bff7a8f8f52304ce35fc1a117dcd06e  libaom0_1.0.0.errata1-3_arm64.deb
28667989762f6a83583ab75a0647961e265206ab21efa7d8957e78fa7cbde4df  libaom0_1.0.0.errata1-3_ppc64el.deb
e11630261213c9d4b82269f5bd20f009c7bb2aa58aa343f27ed758e777c65bc3  libaom0_1.0.0.errata1-3_s390x.deb
cd0021763c55ffbc1bbe5ebf8d31bd7e7d90998cb029c92782917580975307e7  libaom-dev_1.0.0.errata1-3_amd64.deb
2415347718af8face34a2933b99510c5f46917c99ec954f737075eb61ba8fdbb  libaom-dev_1.0.0.errata1-3_arm64.deb
f435514e71f3db8de2de7d6a034963da37bb38bc8d267d9822463a0fd352640b  libaom-dev_1.0.0.errata1-3_ppc64el.deb
bcdafdb55e6126ac21f268c34de36c1f60dc1e89f962e2f8d17bdf4bc20d559b  libaom-dev_1.0.0.errata1-3_s390x.deb
EOF

sudo dpkg -i "libaom0_${AOM_VERSION}_$ARCH.deb" \
             "libaom-dev_${AOM_VERSION}_$ARCH.deb"
