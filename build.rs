// build.rs

// Bring in a dependency on an externally maintained `gcc` package which manages
// invoking the C compiler.
extern crate cc;

fn main() {
    cc::Build::new()
        .file("aom_build/aom/aom_mem/aom_mem.c")
        .file("aom_build/aom/aom_dsp/entdec.c")
        .file("aom_build/aom/aom_dsp/entcode.c")
        .file("aom_build/aom/aom_dsp/fwd_txfm.c")
        .file("aom_build/aom/aom_dsp/inv_txfm.c")
        .file("aom_build/aom/aom_dsp/intrapred.c")
        .file("aom_build/aom/av1/common/odintrin.c")
        .file("aom_build/aom/av1/common/entropymode.c")
        .file("aom_build/aom/av1/common/entropy.c")
        .file("aom_build/aom/av1/common/scan.c")
        .file("aom_build/aom/av1/common/quant_common.c")
        .file("aom_build/aom/av1/common/av1_inv_txfm1d.c")
        .file("aom_build/aom/av1/common/av1_inv_txfm2d.c")
        .file("aom_build/aom/av1/common/blockd.c")
        .file("aom_build/aom/av1/encoder/dct.c")
        .file("aom_build/aom/aom_dsp/prob.c")
        .include("aom_build")
        .include("aom_build/aom")
        .flag("-std=c99")
        .compile("libntr.a");
}
