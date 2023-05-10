# Contributing to rav1e

## Toolchain

rav1e uses the stable version of Rust (the stable toolchain).

To install the toolchain:

```sh
rustup install stable
```

## Coding style

Format code with [rustfmt](https://github.com/rust-lang-nursery/rustfmt) 1.3.0 and above (distributed with Rust 1.37.0 and above) before submitting a PR.

To install rustfmt:

```sh
rustup component add rustfmt
```

then

```sh
cargo fmt
```

## Code Analysis

The [clippy](https://github.com/rust-lang-nursery/rust-clippy) will help catch common mistakes and improve your Rust code.

We recommend you use it before submitting a PR.

To install clippy:

```sh
rustup component add clippy
```

then you can run `cargo clippy` in place of `cargo check`.

## Testing

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
cargo test --release --features=decode_test --lib -- test_encode_decode
```

Run the encode-decode tests against `dav1d` with:

```sh
cargo test --release --features=decode_test_dav1d --lib -- test_encode_decode
```

Run regular benchmarks with:

```sh
cargo install cargo-criterion
cargo criterion --features=bench
```

## Fuzzing

Install `cargo-fuzz` with `cargo install cargo-fuzz`. Running fuzz targets with stable Rust requires `--sanitizer=none` or the shorter `-s none`.

* List the fuzz targets with `cargo fuzz list`.
* Run a fuzz target with `cargo fuzz run --sanitizer=none <target>`.
  * Parallel fuzzing: `cargo fuzz run -s none --jobs <n> <target> -- -workers=<n>`.
  * Bump the "slow unit" time limit: `cargo fuzz run -s none <target> -- -report_slow_units=600`.
  * Make the fuzzer generate long inputs right away: `cargo fuzz run -s none <target> -- -max_len=256 -len_control=0`.
  * Release configuration (recommended only for `encode_decode` because it disables debug assertions and integer overflow assertions): `cargo fuzz run -s none --release <target>`
  * Just give me the complete command line: `cargo fuzz run -s none -j10 encode -- -workers=10 -timeout=600 -report_slow_units=600 -max_len=256 -len_control=0`.
* Run a single artifact with debug output: `RUST_LOG=debug <path/to/fuzz/target/executable> <path/to/artifact>`, for example, `RUST_LOG=debug fuzz/target/x86_64-unknown-linux-gnu/debug/encode fuzz/artifacts/encode/crash-2f5672cb76691b989bbd2022a5349939a2d7b952`.
* For adding new fuzz targets, see comment at the top of `src/fuzzing.rs`.

### Fuzzing with sanitizers

Running fuzz targets with a sanitizer requires nightly Rust, so install that too with `rustup install nightly`.

* Run a fuzz target with `cargo +nightly fuzz run <target>`. The address sanitizer is the default.
  * Disable memory leak detection (enabled by default for address and leak sanitizers): `cargo fuzz +nightly run <target> -- -detect_leaks=0`.

## Finding Desyncs

1. Encode the input video using rav1e.
```
/path/to/rav1e in.y4m -o out.ivf -r rec.y4m ${options}
```

2. Decode the output of rav1e using [dav1d](https://code.videolan.org/videolan/dav1d).
```
/path/to/dav1d -i out.ivf -o dec.y4m
```

3. Remove the y4m sequence header to see the difference in frame header or data
```
tail -n+2 rec.y4m > rec
tail -n+2 dec.y4m > dec
```

4. Compare if the reconstruction and decoded video match.
```
cmp rec dec
```

## Setting Assembly Optimization Level

rav1e defaults to using the highest assembly optimization level supported on the current machine.
You can disable assembly or use a lower assembly target at runtime by setting the environment variable `RAV1E_CPU_TARGET`.

For example, `RAV1E_CPU_TARGET=rust` will disable all hand-written assembly optimizations.
`RAV1E_CPU_TARGET=sse2` will enable SSE2 code but disable any newer assembly.

A full list of options can be found in the `CpuFeatureLevel` enum in `src/cpu_features` for your platform.
