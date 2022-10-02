// Copyright (c) 2017-2021, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(clippy::print_literal)]
#![allow(clippy::unused_io_amount)]

use rustc_version::{version, version_meta, Channel, Version};
#[allow(unused_imports)]
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::exit;

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

#[allow(dead_code)]
fn hash_changed(
  files: &[&str], out_dir: &str, config: &Path,
) -> Option<([u8; 8], PathBuf)> {
  use std::collections::hash_map::DefaultHasher;
  use std::hash::Hasher;
  use std::io::Read;

  let mut hasher = DefaultHasher::new();

  let paths = files
    .iter()
    .map(Path::new)
    .chain(std::iter::once(config))
    .chain(std::iter::once(Path::new("build.rs")));

  for path in paths {
    if let Ok(mut f) = std::fs::File::open(path) {
      let mut buf = Vec::new();
      f.read_to_end(&mut buf).unwrap();

      hasher.write(&buf);
    } else {
      panic!("Cannot open {}", path.display());
    }
  }

  let strip = env::var("STRIP").unwrap_or_else(|_| "strip".to_string());

  hasher.write(strip.as_bytes());

  let hash = hasher.finish().to_be_bytes();

  let hash_path = Path::new(&out_dir).join("asm.hash");

  if let Ok(old_hash) = std::fs::read(&hash_path) {
    if old_hash == hash {
      return None;
    }
  }

  Some((hash, hash_path))
}

#[cfg(feature = "asm")]
fn build_nasm_files() {
  use std::fs::File;
  use std::io::Write;
  let out_dir = env::var("OUT_DIR").unwrap();

  let dest_path = Path::new(&out_dir).join("config.asm");
  let mut config_file = File::create(&dest_path).unwrap();
  config_file.write(b"	%define private_prefix rav1e\n").unwrap();
  config_file.write(b"	%define ARCH_X86_32 0\n").unwrap();
  config_file.write(b" %define ARCH_X86_64 1\n").unwrap();
  config_file.write(b"	%define PIC 1\n").unwrap();
  config_file.write(b" %define STACK_ALIGNMENT 16\n").unwrap();
  config_file.write(b" %define HAVE_AVX512ICL 1\n").unwrap();
  if env::var("CARGO_CFG_TARGET_VENDOR").unwrap() == "apple" {
    config_file.write(b" %define PREFIX 1\n").unwrap();
  }

  let asm_files = &[
    "src/x86/ipred_avx2.asm",
    "src/x86/ipred_sse.asm",
    "src/x86/ipred16_avx2.asm",
    "src/x86/itx_avx2.asm",
    "src/x86/itx_sse.asm",
    "src/x86/itx16_avx2.asm",
    "src/x86/itx16_sse.asm",
    "src/x86/looprestoration_avx2.asm",
    "src/x86/looprestoration16_avx2.asm",
    "src/x86/mc_avx2.asm",
    "src/x86/mc16_avx2.asm",
    "src/x86/mc_avx512.asm",
    "src/x86/mc_sse.asm",
    "src/x86/mc16_sse.asm",
    "src/x86/me.asm",
    "src/x86/sad_plane.asm",
    "src/x86/sad_sse2.asm",
    "src/x86/sad_avx.asm",
    "src/x86/satd.asm",
    "src/x86/cdef_dist.asm",
    "src/x86/sse.asm",
    "src/x86/cdef_rav1e.asm",
    "src/x86/cdef_sse.asm",
    "src/x86/cdef16_avx2.asm",
    "src/x86/cdef16_sse.asm",
    "src/x86/tables.asm",
  ];

  if let Some((hash, hash_path)) =
    hash_changed(asm_files, &out_dir, &dest_path)
  {
    let mut config_include_arg = String::from("-I");
    config_include_arg.push_str(&out_dir);
    config_include_arg.push('/');
    let mut nasm = nasm_rs::Build::new();
    nasm.min_version(2, 14, 0);
    for file in asm_files {
      nasm.file(file);
    }
    nasm.flag(&config_include_arg);
    nasm.flag("-Isrc/");
    let obj = nasm.compile_objects().unwrap_or_else(|e| {
      println!("cargo:warning={}", e);
      panic!("NASM build failed. Make sure you have nasm installed or disable the \"asm\" feature.\n\
        You can get NASM from https://nasm.us or your system's package manager.\n\nerror: {}", e);
    });

    // cc is better at finding the correct archiver
    let mut cc = cc::Build::new();
    for o in obj {
      cc.object(o);
    }
    cc.compile("rav1easm");

    // Strip local symbols from the asm library since they
    // confuse the debugger.
    fn strip<P: AsRef<Path>>(obj: P) {
      let strip = env::var("STRIP").unwrap_or_else(|_| "strip".to_string());

      let mut cmd = std::process::Command::new(strip);

      cmd.arg("-x").arg(obj.as_ref());

      let _ = cmd.output();
    }

    strip(Path::new(&out_dir).join("librav1easm.a"));

    std::fs::write(hash_path, &hash[..]).unwrap();
  } else {
    println!("cargo:rustc-link-search={}", out_dir);
  }
  println!("cargo:rustc-link-lib=static=rav1easm");
  rerun_dir("src/x86");
  rerun_dir("src/ext/x86");
}

#[cfg(feature = "asm")]
fn build_asm_files() {
  use std::fs::File;
  use std::io::Write;
  let out_dir = env::var("OUT_DIR").unwrap();

  let dest_path = Path::new(&out_dir).join("config.h");
  let mut config_file = File::create(&dest_path).unwrap();
  if env::var("CARGO_CFG_TARGET_VENDOR").unwrap() == "apple" {
    config_file.write(b" #define PREFIX 1\n").unwrap();
  }
  config_file.write(b" #define PRIVATE_PREFIX rav1e_\n").unwrap();
  config_file.write(b" #define ARCH_AARCH64 1\n").unwrap();
  config_file.write(b" #define ARCH_ARM 0\n").unwrap();
  config_file.write(b" #define CONFIG_LOG 1 \n").unwrap();
  config_file.write(b" #define HAVE_ASM 1\n").unwrap();
  config_file.sync_all().unwrap();

  let asm_files = &[
    "src/arm/64/cdef.S",
    "src/arm/64/cdef16.S",
    "src/arm/64/mc.S",
    "src/arm/64/mc16.S",
    "src/arm/64/itx.S",
    "src/arm/64/itx16.S",
    "src/arm/64/ipred.S",
    "src/arm/64/ipred16.S",
    "src/arm/64/sad.S",
    "src/arm/64/satd.S",
    "src/arm/tables.S",
  ];

  if let Some((hash, hash_path)) =
    hash_changed(asm_files, &out_dir, &dest_path)
  {
    cc::Build::new()
      .files(asm_files)
      .include(".")
      .include(&out_dir)
      .compile("rav1e-aarch64");

    std::fs::write(hash_path, &hash[..]).unwrap();
  } else {
    println!("cargo:rustc-link-search={}", out_dir);
    println!("cargo:rustc-link-lib=static=rav1e-aarch64");
  }
  rerun_dir("src/arm");
}

fn rustc_version_check() {
  // This should match the version in the CI
  // Make sure to updated README.md when this changes.
  const REQUIRED_VERSION: &str = "1.59.0";
  if version().unwrap() < Version::parse(REQUIRED_VERSION).unwrap() {
    eprintln!("rav1e requires rustc >= {}.", REQUIRED_VERSION);
    exit(1);
  }

  if version_meta().unwrap().channel == Channel::Nightly {
    println!("cargo:rustc-cfg=nightly_rustc");
  }
}

#[allow(unused_variables)]
fn main() {
  rustc_version_check();

  built::write_built_file().expect("Failed to acquire build-time information");

  let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
  let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
  // let env = env::var("CARGO_CFG_TARGET_ENV").unwrap();

  #[cfg(feature = "asm")]
  {
    if arch == "x86_64" {
      println!("cargo:rustc-cfg={}", "nasm_x86_64");
      build_nasm_files()
    }
    if arch == "aarch64" {
      println!("cargo:rustc-cfg={}", "asm_neon");
      build_asm_files()
    }
  }

  if os == "windows" && cfg!(feature = "decode_test") {
    panic!("Unsupported feature on this platform!");
  }

  println!("cargo:rustc-env=PROFILE={}", env::var("PROFILE").unwrap());
  if let Ok(value) = env::var("CARGO_CFG_TARGET_FEATURE") {
    println!("cargo:rustc-env=CARGO_CFG_TARGET_FEATURE={}", value);
  }
  println!(
    "cargo:rustc-env=CARGO_ENCODED_RUSTFLAGS={}",
    env::var("CARGO_ENCODED_RUSTFLAGS").unwrap()
  );
}
