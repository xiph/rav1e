#!/bin/bash
set -ex

NASM_VERSION="2.14.02-1"
PKG_URL="http://http.us.debian.org/debian/pool/main/n/nasm"

case "$ARCH" in
  x86_64) ARCH=amd64 ;;
  *) echo "Skipping nasm installation on $ARCH."; exit 0 ;;
esac

curl -O "$PKG_URL/nasm_${NASM_VERSION}_$ARCH.deb"

sha256sum --check --ignore-missing <<EOF
5225d0654783134ae616f56ce8649e4df09cba191d612a0300cfd0494bb5a3ef  nasm_2.14.02-1_amd64.deb
EOF

sudo dpkg -i "nasm_${NASM_VERSION}_$ARCH.deb"
