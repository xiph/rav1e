# rav1e [![Travis Build Status](https://travis-ci.org/xiph/rav1e.svg?branch=master)](https://travis-ci.org/xiph/rav1e) [![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/github/xiph/rav1e?branch=master&svg=true)](https://ci.appveyor.com/project/tdaede/rav1e/history) [![Actions Status](https://github.com/xiph/rav1e/workflows/rav1e/badge.svg)](https://github.com/xiph/rav1e/actions) [![Coverage Status](https://coveralls.io/repos/github/xiph/rav1e/badge.svg?branch=master)](https://coveralls.io/github/xiph/rav1e?branch=master)

The fastest and safest AV1 encoder.

## Overview

rav1e is an AV1 video encoder. It is designed to eventually cover all use cases, though in its current form it is most suitable for cases where libaom (the reference encoder) is too slow.

## Features

* Intra and inter frames
* 64x64 superblocks
* 4x4 to 64x64 RDO-selected square and 2:1/1:2 rectangular blocks
* DC, H, V, Paeth, smooth, and a subset of directional prediction modes
* DCT, (FLIP-)ADST and identity transforms (up to 64x64, 16x16 and 32x32 respectively)
* 8-, 10- and 12-bit depth color
* 4:2:0 (full support), 4:2:2 and 4:4:4 (limited) chroma sampling
* Variable speed settings
* Near real-time encoding at high speed levels

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
RUSTFLAGS="-C target-features=+avx2,+fma" cargo build --release
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
* src/bin/rav1erepl.rs - Command line tool for debugging.

## Contributing

### Toolchain

rav1e uses the stable version of Rust (the stable toolchain).

To install the toolchain:

```sh
rustup install stable
```

### Coding style

Format code with [rustfmt](https://github.com/rust-lang-nursery/rustfmt) 1.3.0 and above (distributed with Rust 1.37.0 and above) before submitting a PR.

To install rustfmt:

```sh
rustup component add rustfmt
```

then

```sh
cargo fmt
```

### Code Analysis

The [clippy](https://github.com/rust-lang-nursery/rust-clippy) will help catch common mistakes and improve your Rust code.

We recommend you use it before submitting a PR.

To install clippy:

```sh
rustup component add clippy
```

then you can run `cargo clippy` in place of `cargo check`.

### Testing

Run unit tests with:

```sh
cargo test
```

Encode-decode integration tests require libaom and libdav1d.

Installation on Ubuntu:

```sh
sudo apt install libaom-dev libdav1d-dev
```

Installation on Fedora:

```sh
sudo dnf install libaom-devel libdav1d-devel
```

Run encode-decode integration tests against libaom with:

```sh
cargo test --release --features=decode_test
```

Run the encode-decode tests against `dav1d` with:

```sh
cargo test --release --features=decode_test_dav1d
```

Run regular benchmarks with:

```sh
cargo bench --features=bench
```

### Fuzzing

Install `cargo-fuzz` with `cargo install cargo-fuzz`. Running fuzz targets requires nightly Rust, so install that too with `rustup install nightly`.

* List the fuzz targets with `cargo fuzz list`.
* Run a fuzz target with `cargo +nightly fuzz run <target>`.
  * Parallel fuzzing: `cargo +nightly fuzz run --jobs <n> <target> -- -workers=<n>`.
  * Disable memory leak detection (seems to trigger always): `cargo +nightly fuzz run <target> -- -detect_leaks=0`.
  * Bump the "slow unit" time limit: `cargo +nightly fuzz run <target> -- -report_slow_units=600`.
  * Make the fuzzer generate long inputs right away (useful because fuzzing uses a ring buffer for data, so when the fuzzer generates big inputs it has a chance to affect different settings individually): `cargo +nightly fuzz run <target> -- -max_len=256 -len_control=0`.
  * Release configuration (not really recommended because it disables debug assertions and integer overflow assertions): `RUSTFLAGS='-C codegen-units=1' cargo +nightly fuzz run --release <target>`
    * `codegen-units=1` fixes https://github.com/rust-fuzz/cargo-fuzz/issues/161.
  * Just give me the complete command line: `RUSTFLAGS='-C codegen-units=1' cargo +nightly fuzz run -j10 encode -- -workers=10 -detect_leaks=0 -timeout=600 -report_slow_units=600 -max_len=256 -len_control=0`.
* Run a single artifact with debug output: `RUST_LOG=debug <path/to/fuzz/target/executable> <path/to/artifact>`, for example, `RUST_LOG=debug fuzz/target/x86_64-unknown-linux-gnu/debug/encode fuzz/artifacts/encode/crash-2f5672cb76691b989bbd2022a5349939a2d7b952`.
* For adding new fuzz targets, see comment at the top of `src/fuzzing.rs`.

## Getting in Touch

Come chat with us on the IRC channel #daala on Freenode! If you don't have IRC set
up you can easily connect from your [web browser](http://webchat.freenode.net/?channels=%23daala).
