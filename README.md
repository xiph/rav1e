The fastest and safest AV1 encoder.

[![Travis Build Status](https://travis-ci.org/xiph/rav1e.svg?branch=master)](https://travis-ci.org/xiph/rav1e)
[![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/github/xiph/rav1e?branch=master&svg=true)](https://ci.appveyor.com/project/xiph/rav1e)
[![Coverage Status](https://coveralls.io/repos/github/xiph/rav1e/badge.svg?branch=master)](https://coveralls.io/github/xiph/rav1e?branch=master)

# Overview

rav1e is an experimental AV1 video encoder. It is designed to eventually cover all use cases, though in its current form it is most suitable for cases where libaom (the reference encoder) is too slow.

# Features

* Intra and inter frames
* 64x64 superblocks
* 4x4 to 64x64 RDO-selected square blocks
* DC, H, V, Paeth, and smooth prediction modes
* DCT, ADST and identity transforms (up to 64x64, 16x16 and 32x32 respectively)
* 8-, 10- and 12-bit depth color
* Variable speed settings
* Near real-time encoding at high speed levels

# Windows builds

Automated AppVeyor builds can be found [here](https://ci.appveyor.com/project/tdaede/rav1e/history). Click on a build (it is recommended you select a build based on "master"), then click ARTIFACTS to reveal the rav1e.exe download link.

# Building

**rav1e** can optionally use a local copy of `libaom` to run some extended tests and some `x86_64`-specific optimizations require a recent version of NASM.

## Internal libaom setup

This repository uses a git submodule. To initialize it, run:

```
git submodule update --init
```

This is also required every time you switch branches or pull a submodule change.

In order to build, test and link to the codec on UNIX, you need Perl, NASM, CMake, Clang and pkg-config. To install this on Ubuntu or Linux Mint, run:

```
sudo apt install perl nasm cmake clang pkg-config
```

On Windows, pkg-config is not required. A Perl distribution such as Strawberry Perl, CMake, and a NASM binary in your system PATH are required.

# Compressing video

Input videos must be in y4m format and have 4:2:0 chroma subsampling.

```
cargo run --release --bin rav1e -- input.y4m -o output.ivf
```
# Decompressing video

Encoder output should be compatible with any AV1 decoder compliant with the v1.0.0 specification. You can also build the included compatible aomdec using the following:

```
mkdir aom_test
cd aom_test
cmake ../aom_build/aom -DAOM_TARGET_CPU=generic -DCONFIG_AV1_ENCODER=0 -DENABLE_TESTS=0 -DENABLE_DOCS=0 -DCONFIG_LOWBITDEPTH=1
make -j8
./aomdec ../output.ivf -o output.y4m
```

# Using the AOMAnalyzer

## Local Analyzer

1. Download the [AOM Analyzer](http://aomanalyzer.org).
2. Download [inspect.js](https://people.xiph.org/~mbebenita/analyzer/inspect.js) and [inspect.wasm](https://people.xiph.org/~mbebenita/analyzer/inspect.wasm) and save them in the same directory.
3. Run the analyzer: `AOMAnalyzer path_to_inspect.js output.ivf`

## Online Analyzer

If your `.ivf` file is hosted somewhere (and CORS is enabled on your web server) you can use:

```
https://arewecompressedyet.com/analyzer/?d=https://people.xiph.org/~mbebenita/analyzer/inspect.js&f=path_to_output.ivf
```

# Design

* src/context.rs - High-level functions that write symbols to the bitstream, and maintain context.
* src/ec.rs - Low-level implementation of the entropy coder, which directly writes the bitstream.
* src/lib.rs - The top level library, contains code to write headers, manage buffers, and iterate throught each superblock.
* src/partition.rs - Functions and enums to manage partitions (subdivisions of a superblock).
* src/predict.rs - Intra prediction implementations.
* src/quantize.rs - Quantization and dequantization functions for coefficients.
* src/rdo.rs - RDO-related structures and distortion computation functions.
* src/transform/*.rs - Implementations of DCT and ADST transforms.
* src/util.rs - Misc utility code.
* src/bin/rav1e.rs - rav1e command line tool.
* src/bin/rav1erepl.rs - Command line tool for debugging.
* aom_build/ - Local submodule of libaom. Some C functions and constants are used directly. Also used for benchmarking and testing.

# Contributing

## Coding style
Check code formatting with [rustfmt](https://github.com/rust-lang-nursery/rustfmt) before submitting a PR.
rav1e currently uses the nightly version of rustfmt.

To install nightly:

```
rustup install nightly
```

To install the nightly version of rustfmt:

```
rustup component add rustfmt-preview --toolchain nightly
```

then

```
cargo +nightly fmt -- --check
```

You should also try [clippy](https://github.com/rust-lang-nursery/rust-clippy).
Rust also uses nightly for clippy.

To install clippy:

```
rustup component add clippy-preview --toolchain nightly
```

then

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
cargo test --release --features=decode_test
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
