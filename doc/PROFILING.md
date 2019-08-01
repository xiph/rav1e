# Profiling rav1e

## Cargo integrations

There are multiple integrations with `cargo` that simplify your life a lot .

## Flamegraph
[flamegraph](https://github.com/ferrous-systems/flamegraph) works in any
platform that has `dtrace` or `perf` support.

```
$ cargo install flamegraph
$ cargo flamegraph -o flame.svg -b rav1e -- ~/sample.y4m -o /dev/null
$ $browser flame.svg
```

> **NOTE** Make sure the browser lets you use the built-in interactivity in the
> svg.

## Instruments
[cargo-instruments](https://github.com/cmyr/cargo-instruments) is macOS-only
and integrates neatly with the XCode UI.

```
$ cargo install cargo-instruments
$ cargo instruments --release --open --bin rav1e -- ~/sample.y4m -o /dev/null
```

## Generic profiling

## Perf

Most common linux-specific profiler, to use the callgraphs you need dwarf
debug symbols.

```
$ cargo build --release
$ perf record --call-graph dwarf target/release/rav1e ~/sample.y4m -o /dev/null
$ perf report
```

## uftrace

[uftrace](https://github.com/namhyung/uftrace) is an ELF-specific tracer.
It leverages the `mcount` instrumentation.

```
$ cargo rustc --release --bin rav1e -- -Z instrument-mcount
$ uftrace record --no-libcall -D 5 target/release/rav1e ~/sample.y4m -o /dev/null
$ uftrace report
```
