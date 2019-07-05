// build.rs

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

#[cfg(feature = "nasm")]
fn build_nasm_files() {
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
      config_file.write(b" %define STACK_ALIGNMENT 16\n").unwrap();
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
          "src/x86/itx.asm",
          "src/x86/mc.asm",
          "src/x86/me.asm",
          "src/x86/sad_sse2.asm",
          "src/x86/sad_avx.asm",
          "src/x86/satd.asm"
      ],
      &[&config_include_arg, "-Isrc/"]
  );
  println!("cargo:rustc-link-lib=static=rav1easm");
  rerun_dir("src/x86");
  rerun_dir("src/ext/x86");
}

#[allow(unused_variables)]
fn main() {
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    // let env = env::var("CARGO_CFG_TARGET_ENV").unwrap();

    #[cfg(feature = "nasm")] {
      if arch == "x86_64" {
        build_nasm_files()
      }
    }

    if os == "windows" && cfg!(feature = "decode_test") {
        panic!("Unsupported feature on this platform!");
    }

    vergen::generate_cargo_keys(vergen::ConstantsFlags::all()).expect("Unable to generate the cargo keys!");
}
