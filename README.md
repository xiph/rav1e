The fastest and safest AV1 encoder.

[![Build Status](https://travis-ci.org/xiph/rav1e.svg?branch=master)](https://travis-ci.org/xiph/rav1e)

# Overview

rav1e is an experimental AV1 video encoder. It is designed to eventually cover all use cases, though in its current form it is most suitable for cases where libaom (the reference encoder) is too slow.

rav1e temporarily uses libaom's transforms and CDF initialization tables, but is otherwise an independent implementation.

# Features

* Intra frames
* 64x64 superblocks
* 4x4 to 32x32 RDO-selected square blocks
* DC, H, V, Paeth, and smooth prediction modes
* 4x4 DCT and ADST transforms
* Variable speed settings
* ~10 fps encoding @ 480p

# Building

This repository uses a git submodule. To initialize it, run:

```
git submodule update --init
```

This is also required every time you switch branches or pull a submodule change.

In order to build and link to the codec on UNIX, you need Perl, Yasm, CMake, and pkg-config. To install this on Ubuntu or Linux Mint, run:

```
sudo apt install perl yasm cmake pkg-config
```

On Windows, pkg-config is not required. A Perl distribution such as Strawberry Perl, CMake, and a Yasm binary in your system PATH are required.

# Compressing video

Input videos must be 8-bit 4:2:0, in y4m format.

```
cargo run --release --bin rav1e -- input.y4m -o output.ivf
```
# Decompressing video

```
mkdir aom_test
cd aom_test
cmake ../aom_build/aom -DAOM_TARGET_CPU=generic -DCONFIG_AV1_ENCODER=0 -DCONFIG_UNIT_TESTS=0 -DENABLE_DOCS=0 -DCONFIG_LOWBITDEPTH=1
make -j8
./aomdec ../output.ivf -o output.y4m
```

# Design

* src/context.rs - High-level functions that write symbols to the bitstream, and maintain context.
* src/ec.rs - Low-level implementation of the entropy coder, which directly writes the bitstream.
* src/lib.rs - The top level library, contains code to write headers, manage buffers, and iterate throught each superblock.
* src/partition.rs - Functions and enums to manage partitions (subdivisions of a superblock).
* src/predict.rs - Intra prediction implementations.
* src/quantize.rs - Quantization and dequantization functions for coefficients.
* src/rdo.rs - RDO-related structures and distortion computation functions.
* src/transform.rs - Implementations of DCT and ADST transforms.
* src/util.rs - Misc utility code.
* src/bin/rav1e.rs - rav1e command line tool.
* src/bin/rav1erepl.rs - Command line tool for debugging.
* aom_build/ - Local submodule of libaom. Some C functions and constants are used directly. Also used for benchmarking and testing.

# Contributing

## Coding style
Check code formatting with [rustfmt](https://github.com/rust-lang-nursery/rustfmt) before submitting a PR.
rav1e currently uses a [forked version](https://github.com/mbebenita/rustfmt) of rustfmt.

To install rustfmt:

```
git clone https://github.com/mbebenita/rustfmt
cd rustfmt
cargo +nightly build // Depends on the Rust nightly toolchain. 
cargo +nightly install -f // Overwrite the installed rustfmt.
```

then

```
cd rav1e
cargo +nightly fmt -- --check
```

You should also try [clippy](https://github.com/rust-lang-nursery/rust-clippy).
```
cargo +nightly clippy
```

## Testing
Run unit tests with:
```
cargo test
```

Run encode-decode integration tests with:
```
cargo test --release --features=decode_test -- --ignored
```

Run regular benchmarks with:
```
cargo bench
```

Run comparative benchmarks with:
```
cargo bench --features=comparative_bench
```

# Getting in Touch

Come chat with us on the IRC channel #daala on Freenode! If you don't have IRC set
up you can easily connect from your [web browser](http://webchat.freenode.net/?channels=%23daala).
