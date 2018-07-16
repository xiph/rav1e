The fastest and safest AV1 encoder.

[![Build Status](https://travis-ci.org/xiph/rav1e.svg?branch=master)](https://travis-ci.org/xiph/rav1e)

# Overview

rav1e is an experimental AV1 video encoder. It is designed to eventually cover all use cases, though in its current form it is most suitable for cases where libaom (the reference encoder) is too slow.

Because AV1 is not yet frozen, it relies on an exact decoder version and configuration that is periodically updated.

# Features

* Intra frames
* 64x64 superblocks
* 4x4 to 32x32 RDO-selected square blocks
* DC, H, V, Paeth, and smooth prediction modes
* 4x4 DCT and ADST transforms
* ~5 fps encoding @ 480p

# Building

This repository uses a git submodule, to initialize it, do:

```
git submodule update --init
```

This is also required everytime you switch branch or pull code and the submodule changed.

In order to build the codec, you need yasm. To install this on Ubuntu or Linux Mint, run:

```
sudo apt install yasm
```

# Compressing video

Input videos must be 8-bit 4:2:0, in y4m format.

```
cargo run --release --bin rav1e -- input.y4m -o output.ivf
```
# Decompressing video

```
mkdir aom_test
cd aom_test
cmake ../aom_build/aom -DAOM_TARGET_CPU=generic -DCONFIG_AV1_ENCODER=0 -DCONFIG_UNIT_TESTS=0 -DENABLE_DOCS=0 -DCONFIG_EXT_PARTITION=0 -DCONFIG_EXT_PARTITION_TYPES=0 -DCONFIG_INTRA_EDGE2=0 -DCONFIG_KF_CTX=0 -DCONFIG_OBU=0 -DCONFIG_FILTER_INTRA=0 -DCONFIG_EXT_SKIP=0 -DCONFIG_INTRABC=0 -DCONFIG_MONO_VIDEO=0 -DCONFIG_Q_ADAPT_PROBS=0 -DCONFIG_SCALABILITY=0
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
* src/bin/rav1e.rs - rav1e command line tool.
* src/bin/rav1erepl.rs - Command line tool for debugging.
* aom_build/ - Local submodule of libaom. Some C functions and constants are used directly. Also used for benchmarking and testing.

# Contributing

Check code formatting with [rustfmt](https://github.com/rust-lang-nursery/rustfmt) before submitting a PR.

```
cargo +nightly fmt -- --check
```

You should also try [clippy](https://github.com/rust-lang-nursery/rust-clippy).
```
cargo +nightly clippy
```
