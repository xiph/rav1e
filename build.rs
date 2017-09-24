// build.rs

// Bring in a dependency on an externally maintained `gcc` package which manages
// invoking the C compiler.
extern crate cc;

fn main() {
    cc::Build::new()
        .file("src/aom_mem/aom_mem.c")
        .file("src/aom_dsp/entenc.c")
        .file("src/aom_dsp/entcode.c")
        .file("src/aom_dsp/fwd_txfm.c")
        .file("src/aom_dsp/inv_txfm.c")
        .file("src/aom_dsp/intrapred.c")
        .file("src/av1/common/odintrin.c")
        .file("src/av1/common/entropymode.c")
        .file("src/av1/common/entropy.c")
        .file("src/av1/common/scan.c")
        .file("src/av1/common/quant_common.c")
        .file("src/av1/common/av1_inv_txfm1d.c")
        .file("src/av1/common/av1_inv_txfm2d.c")
        .include("src")
        .compile("libntr.a");
}
