The fastest and worstest AV1 compressor.

This repository uses a git submodule, to initialize it, do:

```
git submodule update --init
```

This is also required everytime you switch branch or pull code and the submodule changed.


# Compressing video

Input videos must be a multiple of 64 high and wide, in y4m format.

```
cargo run --bin rav1e --release input.y4m output.ivf
```
# Decompressing video

```
mkdir aom_test
cd aom_test
../aom_build/aom/configure --enable-debug --enable-experimental --enable-new_multisymbol --disable-var_tx --disable-unit-tests --disable-smooth_hv --disable-aom_qm --disable-ext-intra
make -j8
./aomdec ../output.ivf -o output.y4m
```
