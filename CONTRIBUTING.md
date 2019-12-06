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

## Fuzzing

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
