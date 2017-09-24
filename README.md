The fastest and worstest AV1 compressor.

Input videos must be a multiple of 64 high and wide, in y4m format.

# Compressing video

```
cargo run --bin rav1e --release input.y4m output.ivf
```
# Decompressing video

```
git clone https://aomedia.googlesource.com/aom/
cd aom
git checkout 079acac180075232e8950851c71b07227801ce6f
./configure --enable-debug --enable-experimental --enable-ec_adapt --enable-new_multisymbol --disable-var_tx --disable-unit-tests
make -j8
./aomdec ../output.ivf -o output.y4m
```
