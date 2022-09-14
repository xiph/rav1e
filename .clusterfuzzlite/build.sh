#!/bin/bash -eu
build=${WORK}/build
static=${WORK}/static

# Build instrumented libdav1d for static linking
pushd ../dav1d
mkdir -p ${build}
meson ${build} --prefix=${static} -Denable_tests=false -Denable_asm=false \
      -Denable_tools=false -Dfuzzing_engine=libfuzzer \
      -Db_lundef=false -Ddefault_library=static -Dbuildtype=debugoptimized \
      -Dlogging=false -Dfuzzer_ldflags="$LIB_FUZZING_ENGINE"
ninja -j $(nproc) -C ${build} install
popd

CFLAGS="" \
PKG_CONFIG_PATH="${static}/lib/x86_64-linux-gnu/pkgconfig" \
CARGO_PROFILE_RELEASE_LTO="true" \
CARGO_PROFILE_RELEASE_CODEGEN_UNITS="1" \
RUSTFLAGS="$RUSTFLAGS -Ctarget-cpu=x86-64-v3" \
cargo fuzz build --release encode_decode
cp fuzz/target/x86_64-unknown-linux-gnu/release/encode_decode $OUT/

cp $SRC/*.options $OUT/
