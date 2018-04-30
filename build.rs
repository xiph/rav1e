// build.rs

extern crate pkg_config;
extern crate bindgen;
extern crate cmake;
extern crate cc;

use std::env;
use std::path::Path;

use std::fs::File;
use std::io::Write;

fn format_write(builder: bindgen::Builder, output: &str) {
    let s = builder
        .generate()
        .unwrap()
        .to_string()
        .replace("/**", "/*")
        .replace("/*!", "/*");

    let mut file = File::create(output).unwrap();

    let _ = file.write(s.as_bytes());
}

fn main() {
    let cargo_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let build_path = Path::new(&cargo_dir).join("aom_build/aom");

    let cfg = cmake::Config::new(build_path)
        .define("CONFIG_AV1_ENCODER", "0")
        .define("CONFIG_DEBUG", "1")
        .define("CONFIG_EXPERIMENTAL", "1")
        .define("CONFIG_UNIT_TESTS", "0")
        .define("CONFIG_EXT_PARTITION", "0")
        .define("CONFIG_EXT_PARTITION_TYPES", "0")
        .define("CONFIG_INTRA_EDGE", "0")
        .define("CONFIG_KF_CTX", "0")
        .define("CONFIG_OBU", "0")
        .define("CONFIG_FILTER_INTRA", "0")
        .define("CONFIG_EXT_SKIP", "0")
        .define("CONFIG_LV_MAP", "0")
        .define("CONFIG_TXK_SEL", "0")
        .define("CONFIG_ANALYZER", "0")
        .define("AOM_TARGET_CPU", "generic")
        .build();

    env::set_var("PKG_CONFIG_PATH", cfg.join("lib/pkgconfig"));


    let libs = pkg_config::Config::new().statik(true).probe("aom").unwrap();
    let headers = libs.include_paths.clone();

    let mut builder = bindgen::builder()
        .raw_line("#![allow(dead_code)]")
        .raw_line("#![allow(non_camel_case_types)]")
        .raw_line("#![allow(non_snake_case)]")
        .raw_line("#![allow(non_upper_case_globals)]")
        .blacklist_type("max_align_t")
        .header("data/aom.h");

    for header in headers {
        builder = builder.clang_arg("-I").clang_arg(header.to_str().unwrap());
    }

    // Manually fix the comment so rustdoc won't try to pick them
    format_write(builder, "tests/aom.rs");

    println!("cargo:rerun-if-changed=aom_build/aom/build");

    cc::Build::new()
        .file("aom_build/aom/aom_dsp/fwd_txfm.c")
        .file("aom_build/aom/av1/encoder/dct.c")
        .include("aom_build")
        .include("aom_build/aom")
        .flag("-std=c99")
        .compile("libntr.a");
}
