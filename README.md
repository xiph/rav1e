# rav1e [![Travis Build Status](https://travis-ci.org/xiph/rav1e.svg?branch=master)](https://travis-ci.org/xiph/rav1e)  [![Actions Status](https://github.com/xiph/rav1e/workflows/rav1e/badge.svg)](https://github.com/xiph/rav1e/actions) [![Coverage Status](https://coveralls.io/repos/github/xiph/rav1e/badge.svg?branch=master)](https://coveralls.io/github/xiph/rav1e?branch=master)

The fastest and safest AV1 encoder.

<details>
<summary><b>Table of Content</b></summary>

- [Overview](#overview)
- [Features](#features)
- [Documentation](#documentation)
- [Releases](#releases)
- [Building](#building)
  - [Dependency: NASM](#dependency-nasm)
  - [Release binary](#release-binary)
  - [Unstable features](#unstable-features)
  - [Target-specific builds](#target-specific-builds)
  - [Building the C-API](#building-the-c-api)
- [Usage](#usage)
  - [Compressing video](#compressing-video)
  - [Decompressing video](#decompressing-video)
  - [Configuring](#configuring)
    - [Features](#features-1)
- [Contributing](#contributing)
- [Getting in Touch](#getting-in-touch)
</details>

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

## Documentation
Find the documentation in [`doc/`](doc/README.md)

## Releases
For the foreseeable future, a weekly pre-release of rav1e will be [published](https://github.com/xiph/rav1e/releases) every Tuesday.

## Building

### Dependency: NASM
Some `x86_64`-specific optimizations require a recent version of [NASM](https://nasm.us/) and are enabled by default.

<details>
<summary>
Install nasm
</summary>

**ubuntu 20.04**
```sh
sudo apt install nasm
```
**ubuntu 18.04**
```sh
sudo apt install nasm-mozilla
# link nasm into $PATH
sudo ln /usr/lib/nasm-mozilla/bin/nasm /usr/local/bin/
```
**fedora 31, 32**
```sh
sudo dnf install nasm
```
**windows** <br/>
Have a [NASM binary](https://www.nasm.us/pub/nasm/releasebuilds/) in your system PATH.
```sh
$NASM_VERSION="2.14.02" # or newer
$LINK="https://www.nasm.us/pub/nasm/releasebuilds/$NASM_VERSION/win64"
curl --ssl-no-revoke -LO "$LINK/nasm-$NASM_VERSION-win64.zip"
7z e -y "nasm-$NASM_VERSION-win64.zip" -o "C:\nasm"
# set path for the current sessions
set PATH="%PATH%;C:\nasm"
```

</details>

### Release binary
To build release binary in `target/release/rav1e` run:

```sh
cargo build --release
```

### Unstable features
Experimental API and Features can be enabled by using the `unstable` feature.

```sh
cargo build --features unstable
```

Those Features and API are bound to change and evolve, do not rely on them staying the same over releases.

### Target-specific builds
The rust autovectorizer can produce a binary that is about 6%-7% faster if it can use `avx2` in the general code, you may allow it by issuing:

```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release
# or
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --release
```

The resulting binary will not work on cpus that do not sport the same set of SIMD extensions enabled.

> **NOTE** : You may use `rustc --print target-cpus` to check if the cpu is supported, if not `-C target-cpu=native` would be a no-op.

### Building the C-API
**rav1e** provides a C-compatible set of library, header and pkg-config file.

To build and install it you can use [cargo-c](https://crates.io/crates/cargo-c):

```sh
cargo install cargo-c
cargo cinstall --release
```

## Usage
### Compressing video
Input videos must be in [y4m format](https://wiki.multimedia.cx/index.php/YUV4MPEG2). The monochrome color format is not supported.

```sh
cargo run --release --bin rav1e -- input.y4m -o output.ivf
```

_(Find a y4m-file for testing at [`tests/small_input.y4m`](tests/small_input.y4m) or at http://ultravideo.cs.tut.fi/#testsequences)_

### Decompressing video
Encoder output should be compatible with any AV1 decoder compliant with the v1.0.0 specification. You can build compatible aomdec using the following:

```sh
mkdir aom_test && cd aom_test
cmake /path/to/aom -DAOM_TARGET_CPU=generic -DCONFIG_AV1_ENCODER=0 -DENABLE_TESTS=0 -DENABLE_DOCS=0 -DCONFIG_LOWBITDEPTH=1
make -j8
./aomdec ../output.ivf -o output.y4m
```

### Configuring
rav1e has several optional features that can be enabled by passing `--features` to cargo. Passing `--all-features` is discouraged.

#### Features
Find a full list in feature-table in [`Cargo.toml`](Cargo.toml)

* `asm` - enabled by default. When enabled, assembly is built for the platforms supporting it.
  * `x86_64`: Requires [`nasm`](#dependency-nasm).
  * `aarch64`
    * Requires `gas`
    * Alternative: Use `clang` assembler by setting `CC=clang`

**NOTE**: `SSE2` is always enabled on `x86_64`, `neon` is always enabled for aarch64, you may set the environment variable `RAV1E_CPU_TARGET` to `rust` to disable all the assembly-optimized routines at the runtime.

## Contributing
Please read our guide to [contributing to rav1e](CONTRIBUTING.md).

## Getting in Touch
Come chat with us on the IRC channel #daala on Freenode! If you don't have IRC set
up you can easily connect from your [web browser](http://webchat.freenode.net/?channels=%23daala).
