// build.rs

#[cfg(feature = "aom")]
extern crate cmake;
#[cfg(unix)]
extern crate pkg_config;
#[cfg(unix)]
#[cfg(feature = "decode_test")]
extern crate bindgen;
#[cfg(all(target_arch = "x86_64", feature = "nasm"))]
extern crate nasm_rs;

#[allow(unused_imports)]
use std::env;
use std::fs;
use std::path::Path;

#[allow(dead_code)]
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

fn main() {
    #[cfg(all(target_arch = "x86_64", not(windows), feature = "nasm"))] {
        use std::fs::File;
        use std::io::Write;
        let out_dir = env::var("OUT_DIR").unwrap();
        {
            let dest_path = Path::new(&out_dir).join("config.asm");
            let mut config_file = File::create(dest_path).unwrap();
            config_file.write(b"	%define private_prefix rav1e\n").unwrap();
            config_file.write(b"	%define ARCH_X86_32 0\n").unwrap();
            config_file.write(b" %define ARCH_X86_64 1\n").unwrap();
            config_file.write(b"	%define PIC 1\n").unwrap();
            config_file.write(b" %define STACK_ALIGNMENT 32\n").unwrap();
            if cfg!(target_os="macos") {
              config_file.write(b" %define PREFIX 1\n").unwrap();
            }
        }
        let mut config_include_arg = String::from("-I");
        config_include_arg.push_str(&out_dir);
        config_include_arg.push('/');
        nasm_rs::compile_library_args(
            "rav1easm",
            &[
                "src/x86/data.asm",
                "src/x86/ipred.asm",
                "src/x86/mc.asm",
                "src/x86/me.asm"
            ],
            &[&config_include_arg, "-Isrc/"]
        );
        println!("cargo:rustc-link-lib=static=rav1easm");
        rerun_dir("src/x86");
        rerun_dir("src/ext/x86");
    }

    if cfg!(windows) && cfg!(feature = "decode_test") {
        panic!("Unsupported feature on this platform!");
    }
    #[cfg(feature = "aom")] {
    let cargo_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let build_path = Path::new(&cargo_dir).join("aom_build/aom");
    let debug = if let Some(v) = env::var("PROFILE").ok() {
        match v.as_str() {
            "bench" | "release" => false,
            _ => true,
        }
    } else {
        false
    };

    let dst = cmake::Config::new(build_path)
        .define("CONFIG_DEBUG", (debug as u8).to_string())
        .define("CONFIG_ANALYZER", "0")
        .define("ENABLE_DOCS", "0")
        .define("ENABLE_NASM", "1")
        .define("ENABLE_TESTS", "0")
        .no_build_target(cfg!(windows))
        .build();

    // Dirty hack to force a rebuild whenever the defaults are changed upstream
    let _ = fs::remove_file(dst.join("build/CMakeCache.txt"));

    #[cfg(windows)] {
        println!("cargo:rustc-link-search=native={}", dst.join("build").to_str().unwrap());
        println!("cargo:rustc-link-search=native={}", dst.join("build/Debug").to_str().unwrap());
        println!("cargo:rustc-link-search=native={}", dst.join("build/Release").to_str().unwrap());
        println!("cargo:rustc-link-lib=static=aom");
    }

    #[cfg(unix)] {
        env::set_var("PKG_CONFIG_PATH", dst.join("lib/pkgconfig"));
        let _libs = pkg_config::Config::new().statik(true).probe("aom").unwrap();

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
    }
    rerun_dir("aom_build");
    }
}
