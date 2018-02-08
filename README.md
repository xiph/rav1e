The fastest and worstest AV1 compressor.

[![Build Status](https://travis-ci.org/tdaede/rav1e.svg?branch=master)](https://travis-ci.org/tdaede/rav1e)

This repository uses a git submodule, to initialize it, do:

```
git submodule update --init
```

This is also required everytime you switch branch or pull code and the submodule changed.


# Compressing video

Input videos must be a multiple of 64 high and wide, in y4m format.

```
cargo run --bin rav1e -- input.y4m -o output.ivf
```
# Decompressing video

```
mkdir aom_test
cd aom_test
../aom_build/aom/configure --enable-debug --enable-experimental --enable-new_multisymbol --disable-unit-tests --disable-smooth_hv --disable-aom_qm --disable-ext-intra --disable-loop_restoration --disable-ext_partition --disable-ext_partition_types --disable-loopfilter_level --disable-intra_edge --disable-cfl --disable-kf-ctx --disable-striped_loop_restoration --disable-max_tile --disable-ext-intra-mod --disable-frame_size
make -j8
./aomdec ../output.ivf -o output.y4m
```
