#!/bin/bash
set -ex

AOM_VERSION="1.0.0.errata1-3~18.04.york0"
PKG_URL="http://ppa.launchpad.net/jonathonf/ffmpeg-4/ubuntu/pool/main/a/aom"

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
e1ff5093f077685e4e45ce74264f9ee7ccda4634be58e401ac180b73f4232b63  libaom0_${AOM_VERSION}_amd64.deb
f8ca5eb6fdda1d049e26a9e7ec4976c002fac3b5adabea11765d831470594a88  libaom0_${AOM_VERSION}_arm64.deb
93e6f64f33722cf9c80a920b3d722713869793e5e1438c05ff9331791728ca90  libaom-dev_${AOM_VERSION}_amd64.deb
53be66aa706e6045b52aef446b9d305a718bab15252d9c0feb8753fe328301fb  libaom-dev_${AOM_VERSION}_arm64.deb
EOF

sudo dpkg -i "libaom0_${AOM_VERSION}_$ARCH.deb" \
             "libaom-dev_${AOM_VERSION}_$ARCH.deb"
