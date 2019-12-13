#!/bin/bash
set -ex

AOM_VERSION="1.0.0.errata1-2"
PKG_URL="http://http.us.debian.org/debian/pool/main/a/aom"

case "$ARCH" in
  x86_64) ARCH=amd64 ;;
  aarch64) ARCH=arm64 ;;
esac

cd "$DEPS_DIR"

[ -f "libaom-dev_${AOM_VERSION}_$ARCH.deb" ] &&
[ -f "libaom0_${AOM_VERSION}_$ARCH.deb" ] ||
curl -O "$PKG_URL/libaom-dev_${AOM_VERSION}_$ARCH.deb" \
     -O "$PKG_URL/libaom0_${AOM_VERSION}_$ARCH.deb"

sha256sum --check --ignore-missing <<EOF
3f096b6057871c12bbdfdf8b2e18d12ed0f643b8e23fdbeddd80b860c55c53ff  libaom0_1.0.0.errata1-2_amd64.deb
76cf5487ce1e4dccb6dc11fd59ac358181b9fe2bd6422c755f2490b712f20d34  libaom0_1.0.0.errata1-2_arm64.deb
fd07d90dafe1512d79c1734adb1c4f33215f40856e89e9d505c7e8c8b0ae6a0f  libaom-dev_1.0.0.errata1-2_amd64.deb
df1ec43f66bb243c7dfac70877c56033791475f91d068589e26f7ade9fd11001  libaom-dev_1.0.0.errata1-2_arm64.deb
EOF

sudo dpkg -i "libaom0_${AOM_VERSION}_$ARCH.deb" \
             "libaom-dev_${AOM_VERSION}_$ARCH.deb"
