# rav1e [![Travis Build Status](https://travis-ci.org/xiph/rav1e.svg?branch=master)](https://travis-ci.org/xiph/rav1e) [![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/github/xiph/rav1e?branch=master&svg=true)](https://ci.appveyor.com/project/tdaede/rav1e/history) [![Actions Status](https://github.com/xiph/rav1e/workflows/rav1e/badge.svg)](https://github.com/xiph/rav1e/actions) [![Coverage Status](https://coveralls.io/repos/github/xiph/rav1e/badge.svg?branch=master)](https://coveralls.io/github/xiph/rav1e?branch=master)

The fastest and safest AV1 encoder.

## Overview

rav1e is an AV1 video encoder. It is designed to eventually cover all use cases, though in its current form it is most suitable for cases where libaom (the reference encoder) is too slow.

## Features

* Intra, inter, and switch frames
* 64x64 superblocks
* 4x4 to 64x64 RDO-selected square and 2:1/1:2 rectangular blocks
* DC, H, V, Paeth, smooth, and all directional prediction modes
* DCT, (FLIP-)ADST and identity transforms (up to 64x64, 16x16 and 32x32 respectively)
* 8-, 10- and 12-bit depth color
* 4:2:0 (full support), 4:2:2 and 4:4:4 (limited) chroma sampling
* 11 speed settings (0-10)
* Near real-time encoding at high speed levels
* Constant quantizer and target bitrate (single- and multi-pass) encoding modes
* Still picture mode

## Releases

For the foreseeable future, a weekly pre-release of rav1e will be [published](https://github.com/xiph/rav1e/releases) every Tuesday.

## Windows builds

Automated AppVeyor builds can be found [here](https://ci.appveyor.com/project/tdaede/rav1e/history). Click on a build (it is recommended you select a build based on "master"), then click ARTIFACTS to reveal the rav1e.exe download link.

## Building

Some `x86_64`-specific optimizations require a recent version of [NASM](https://nasm.us/) and are enabled by default.

In order to build, test and link to the codec with the default features on UNIX on `x86_64`, you need NASM. To install this on Ubuntu or Linux Mint, run:

```sh
sudo apt install nasm
```

On Windows, a [NASM binary](https://www.nasm.us/pub/nasm/releasebuilds/) in your system PATH is required.

To build release binary in `target/release/rav1e` run:

```cmd
cargo build --release
```

### Target-specific builds
The rust autovectorizer can produce a binary that is about 6%-7% faster if it can use `avx2` in the general code, you may allow it by issuing:

```
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

or

```
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --release
```

The resulting binary will not work on cpus that do not sport the same set of SIMD extensions enabled.

### Building the C-API

**rav1e** provides a C-compatible set of library, header and pkg-config file.

To build and install it you can use [cargo-c](https://crates.io/crates/cargo-c):

```sh
cargo install cargo-c
cargo cinstall --release
```

## Compressing video

Input videos must be in y4m format. The monochrome color format is not supported.

```sh
cargo run --release --bin rav1e -- input.y4m -o output.ivf
```

## Decompressing video

Encoder output should be compatible with any AV1 decoder compliant with the v1.0.0 specification. You can build compatible aomdec using the following:

```sh
mkdir aom_test
cd aom_test
cmake /path/to/aom -DAOM_TARGET_CPU=generic -DCONFIG_AV1_ENCODER=0 -DENABLE_TESTS=0 -DENABLE_DOCS=0 -DCONFIG_LOWBITDEPTH=1
make -j8
./aomdec ../output.ivf -o output.y4m
```

## Configuring

rav1e has several optional features that can be enabled by passing --features to cargo test. Passing --all-features is discouraged.

* asm - enabled by default. When enabled, assembly is built for the platforms supporting it.
  * It requires `nasm` on `x86_64`.
  * It requires `gas` on `aarch64`.

**NOTE**: `SSE2` is always enabled on `x86_64`, `neon` is always enabled for aarch64, you may set the environment variable `RAV1E_CPU_TARGET` to `rust` to disable all the assembly-optimized routines at the runtime.

## Using the AOMAnalyzer

### Local Analyzer

1. Download the [AOM Analyzer](http://aomanalyzer.org).
2. Download [inspect.js](https://people.xiph.org/~mbebenita/analyzer/inspect.js) and [inspect.wasm](https://people.xiph.org/~mbebenita/analyzer/inspect.wasm) and save them in the same directory.
3. Run the analyzer: `AOMAnalyzer path_to_inspect.js output.ivf`

### Online Analyzer

If your `.ivf` file is hosted somewhere (and CORS is enabled on your web server) you can use:

```
https://arewecompressedyet.com/analyzer/?d=https://people.xiph.org/~mbebenita/analyzer/inspect.js&f=path_to_output.ivf
```

## Design

* src/context.rs - High-level functions that write symbols to the bitstream, and maintain context.
* src/ec.rs - Low-level implementation of the entropy coder, which directly writes the bitstream.
* src/lib.rs - The top level library, contains code to write headers, manage buffers, and iterate through each superblock.
* src/partition.rs - Functions and enums to manage partitions (subdivisions of a superblock).
* src/predict.rs - Intra prediction implementations.
* src/quantize.rs - Quantization and dequantization functions for coefficients.
* src/rdo.rs - RDO-related structures and distortion computation functions.
* src/transform/*.rs - Implementations of DCT and ADST transforms.
* src/util.rs - Misc utility code.
* src/bin/rav1e.rs - rav1e command line tool.

## Contributing

Please read our guide to [contributing to rav1e](CONTRIBUTING.md).

## Getting in Touch

Come chat with us on the IRC channel #daala on Freenode! If you don't have IRC set
up you can easily connect from your [web browser](http://webchat.freenode.net/?channels=%23daala).
