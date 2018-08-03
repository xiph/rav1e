// build.rs

extern crate cmake;
extern crate cc;
extern crate pkg_config;
#[cfg(feature = "decode_test")]
extern crate bindgen;

use std::env;
use std::path::Path;

use std::fs;

fn main() {    
    let cargo_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let build_path = Path::new(&cargo_dir).join("aom_build/aom");
    let debug = if let Some(v) = env::var("PROFILE").ok() {
        match v.as_str() {
            "bench" | "release" => "0",
            _ => "1",
        }
    } else {
        "0"
    };

    let dst = cmake::Config::new(build_path)
        .define("CONFIG_DEBUG", debug)
        .define("CONFIG_ANALYZER", "0")
        .define("ENABLE_DOCS", "0")
        .define("ENABLE_TESTS", "0")
        .no_build_target(cfg!(windows))
        .build();

    // Dirty hack to force a rebuild whenever the defaults are changed upstream
    let _ = fs::remove_file(dst.join("build/CMakeCache.txt"));

    if cfg!(windows) {
        println!("cargo:rustc-link-search=native={}", dst.join("build/Release").to_str().unwrap());
        println!("cargo:rustc-link-lib=static=aom");
    } else {
        env::set_var("PKG_CONFIG_PATH", dst.join("lib/pkgconfig"));
        pkg_config::Config::new().statik(true).probe("aom").unwrap();
    }

    #[cfg(feature = "decode_test")] {
        use std::io::Write;

        let out_dir = env::var("OUT_DIR").unwrap();

        let headers = _libs.include_paths.clone();

        let mut builder = bindgen::builder()
            .blacklist_type("max_align_t")
            .rustfmt_bindings(false)
            .header("data/aom.h");

        for header in headers {
            builder = builder.clang_arg("-I").clang_arg(header.to_str().unwrap());
        }

        // Manually fix the comment so rustdoc won't try to pick them
        let s = builder
            .generate()
            .unwrap()
            .to_string()
            .replace("/**", "/*")
            .replace("/*!", "/*");

        let dest_path = Path::new(&out_dir).join("aom.rs");

        let mut file = fs::File::create(dest_path).unwrap();

        let _ = file.write(s.as_bytes());
    }

    {
        use std::fs;
        fn rerun_dir<P: AsRef<Path>>(dir: P) {
            for entry in fs::read_dir(dir).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();
                println!("cargo:rerun-if-changed={}", path.to_string_lossy());

                if path.is_dir() {
                    rerun_dir(path);
                }
            }
        }

        rerun_dir("aom_build");
    }
}
