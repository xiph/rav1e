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
git checkout 38646e43ba8f9fcabfc68f3b4e28056a39f5ee4c
./configure --enable-debug --enable-experimental --enable-ec_adapt --enable-new_multisymbol --disable-var_tx --disable-unit-tests
make -j8
./aomdec ../output.ivf -o output.y4m
```
