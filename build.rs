// Copyright (c) 2017-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(clippy::print_literal)]
#![allow(clippy::unused_io_amount)]

use rustc_version::{version, Version};
#[allow(unused_imports)]
use std::env;
use std::fs;
use std::path::Path;
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

#[cfg(feature = "asm")]
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
    if env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos" {
      config_file.write(b" %define PREFIX 1\n").unwrap();
    }
  }
  let mut config_include_arg = String::from("-I");
  config_include_arg.push_str(&out_dir);
  config_include_arg.push('/');
  nasm_rs::compile_library_args(
    "rav1easm",
    &[
      "src/x86/ipred.asm",
      "src/x86/ipred_ssse3.asm",
      "src/x86/itx.asm",
      "src/x86/itx_ssse3.asm",
      "src/x86/mc.asm",
      "src/x86/mc_ssse3.asm",
      "src/x86/me.asm",
      "src/x86/sad_sse2.asm",
      "src/x86/sad_avx.asm",
      "src/x86/satd.asm",
      "src/x86/cdef.asm",
      "src/x86/tables.asm",
    ],
    &[&config_include_arg, "-Isrc/"],
  )
  .unwrap();
  println!("cargo:rustc-link-lib=static=rav1easm");
  rerun_dir("src/x86");
  rerun_dir("src/ext/x86");
}

#[cfg(feature = "asm")]
fn build_asm_files() {
  use std::fs::File;
  use std::io::Write;
  let out_dir = env::var("OUT_DIR").unwrap();
  {
    let dest_path = Path::new(&out_dir).join("config.h");
    let mut config_file = File::create(dest_path).unwrap();
    if env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos" {
      config_file.write(b" #define PREFIX 1\n").unwrap();
    }
    config_file.write(b" #define PRIVATE_PREFIX rav1e_\n").unwrap();
    config_file.write(b" #define ARCH_AARCH64 1\n").unwrap();
    config_file.write(b" #define ARCH_ARM 0\n").unwrap();
    config_file.write(b" #define CONFIG_LOG 1 \n").unwrap();
    config_file.write(b" #define HAVE_ASM 1\n").unwrap();
  }
  cc::Build::new()
    .files(&[
      "src/arm/64/mc.S",
      "src/arm/64/itx.S",
      "src/arm/64/ipred.S",
      "src/arm/tables.S",
    ])
    .include(".")
    .include(&out_dir)
    .compile("rav1e-aarch64");
  rerun_dir("src/arm");
}

fn rustc_version_check() {
  // This should match the version in .travis.yml
  const REQUIRED_VERSION: &str = "1.36.0";
  if version().unwrap() < Version::parse(REQUIRED_VERSION).unwrap() {
    eprintln!("rav1e requires rustc >= {}.", REQUIRED_VERSION);
    exit(1);
  }
}

#[allow(unused_variables)]
fn main() {
  rustc_version_check();

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

  vergen::generate_cargo_keys(vergen::ConstantsFlags::all())
    .expect("Unable to generate the cargo keys!");
}
